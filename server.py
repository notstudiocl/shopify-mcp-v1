#!/usr/bin/env python3
"""
Shopify MCP Server — Full Admin API access via FastMCP.
Provides tools for managing products, orders, customers, collections,
inventory, and fulfillments through the Shopify Admin REST API.

Token Management:
  - Uses client_credentials grant to auto-generate and refresh tokens
  - Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (recommended for OAuth apps)
  - Falls back to static SHOPIFY_ACCESS_TOKEN if client credentials not set
"""
import json
import os
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_STORE        = os.environ.get("SHOPIFY_STORE", "")           # e.g. "my-store"
SHOPIFY_TOKEN        = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")    # Static token (shpat_...)
SHOPIFY_CLIENT_ID    = os.environ.get("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET = os.environ.get("SHOPIFY_CLIENT_SECRET", "")
API_VERSION          = os.environ.get("SHOPIFY_API_VERSION", "2024-10")

# Refresh buffer: refresh token 30 minutes before expiry (only used with OAuth)
TOKEN_REFRESH_BUFFER = int(os.environ.get("TOKEN_REFRESH_BUFFER", "1800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shopify_mcp")

PORT          = int(os.environ.get("PORT", "8000"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

mcp = FastMCP("shopify_mcp", host="0.0.0.0", port=PORT, json_response=True)


# ---------------------------------------------------------------------------
# Token Manager — handles automatic token lifecycle
# ---------------------------------------------------------------------------

class TokenManager:
    """
    Manages Shopify Admin API access tokens.

    Two modes:
      1. Static token  — set SHOPIFY_ACCESS_TOKEN (recommended for Custom Apps)
      2. OAuth / client_credentials — set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET
         Enables auto-refresh before expiry and retry on 401.
    """

    def __init__(
        self,
        store: str,
        client_id: str,
        client_secret: str,
        static_token: str = "",
        refresh_buffer: int = 1800,
    ):
        self._store         = store
        self._client_id     = client_id
        self._client_secret = client_secret
        self._static_token  = static_token
        self._refresh_buffer = refresh_buffer

        self._access_token: str   = ""
        self._expires_at: float   = 0.0
        self._lock = asyncio.Lock()

        self._use_client_credentials = bool(client_id and client_secret)

        if self._use_client_credentials:
            logger.info("Token mode: client_credentials (auto-refresh enabled)")
        elif static_token:
            logger.info("Token mode: static SHOPIFY_ACCESS_TOKEN (no auto-refresh)")
            self._access_token = static_token
            self._expires_at   = float("inf")
        else:
            logger.warning(
                "No credentials configured. Set SHOPIFY_ACCESS_TOKEN or "
                "SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET."
            )

    @property
    def is_expired(self) -> bool:
        if not self._access_token:
            return True
        return time.time() >= (self._expires_at - self._refresh_buffer)

    async def get_token(self) -> str:
        if not self.is_expired:
            return self._access_token

        async with self._lock:
            if not self.is_expired:
                return self._access_token

            if self._use_client_credentials:
                await self._refresh_token()
            elif not self._access_token:
                raise RuntimeError(
                    "No valid token available. "
                    "Set SHOPIFY_ACCESS_TOKEN in your environment variables."
                )

        return self._access_token

    async def force_refresh(self) -> str:
        if not self._use_client_credentials:
            raise RuntimeError(
                "Cannot refresh — using a static token. "
                "Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET to enable auto-refresh."
            )
        async with self._lock:
            await self._refresh_token()
        return self._access_token

    async def _refresh_token(self) -> None:
        url = f"https://{self._store}.myshopify.com/admin/oauth/access_token"
        logger.info("Refreshing Shopify access token via client_credentials grant...")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )

            if resp.status_code != 200:
                logger.error(f"Token refresh failed ({resp.status_code}): {resp.text[:500]}")
                raise RuntimeError(
                    f"Token refresh failed ({resp.status_code}). "
                    "Check SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET."
                )

            data               = resp.json()
            self._access_token = data["access_token"]
            expires_in         = data.get("expires_in", 86399)
            self._expires_at   = time.time() + expires_in

            scope         = data.get("scope", "")
            scope_preview = scope[:80] + "..." if len(scope) > 80 else scope
            logger.info(
                f"Token refreshed. Expires in {expires_in}s "
                f"({expires_in // 3600}h {(expires_in % 3600) // 60}m). "
                f"Scopes: {scope_preview}"
            )


# Global token manager
token_manager = TokenManager(
    store=SHOPIFY_STORE,
    client_id=SHOPIFY_CLIENT_ID,
    client_secret=SHOPIFY_CLIENT_SECRET,
    static_token=SHOPIFY_TOKEN,
    refresh_buffer=TOKEN_REFRESH_BUFFER,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return f"https://{SHOPIFY_STORE}.myshopify.com/admin/api/{API_VERSION}"


async def _headers() -> dict:
    token = await token_manager.get_token()
    return {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }


async def _request(
    method: str,
    path: str,
    params: Optional[dict] = None,
    body:   Optional[dict] = None,
    _retried: bool = False,
) -> dict:
    """Central HTTP helper — every API call flows through here.
    Auto-retries once on 401 when using OAuth credentials.
    """
    if not SHOPIFY_STORE:
        raise RuntimeError(
            "Missing SHOPIFY_STORE environment variable. "
            "Set it before starting the server."
        )

    url     = f"{_base_url()}/{path}"
    headers = await _headers()

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method, url,
            headers=headers,
            params=params,
            json=body,
            timeout=30.0,
        )

        if resp.status_code == 401 and not _retried and token_manager._use_client_credentials:
            logger.warning("Got 401 — refreshing token and retrying...")
            await token_manager.force_refresh()
            return await _request(method, path, params=params, body=body, _retried=True)

        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()


def _error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        messages = {
            401: "Authentication failed — check your SHOPIFY_ACCESS_TOKEN (should start with shpat_).",
            403: "Permission denied — your token may be missing required API scopes.",
            404: "Resource not found — double-check the ID.",
            422: f"Validation error: {json.dumps(detail)}",
            429: "Rate-limited — wait a moment and retry.",
        }
        return messages.get(status, f"Shopify API error {status}: {json.dumps(detail)}")
    if isinstance(e, httpx.TimeoutException):
        return "Request timed out — try again."
    if isinstance(e, RuntimeError):
        return str(e)
    return f"Unexpected error: {type(e).__name__}: {e}"


def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════

class ListProductsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int]  = Field(default=50, ge=1, le=250, description="Max products to return (1-250)")
    status:         Optional[str]  = Field(default=None, description="Filter by status: active, archived, draft")
    product_type:   Optional[str]  = Field(default=None, description="Filter by product type")
    vendor:         Optional[str]  = Field(default=None, description="Filter by vendor name")
    collection_id:  Optional[int]  = Field(default=None, description="Filter by collection ID")
    since_id:       Optional[int]  = Field(default=None, description="Pagination: return products after this ID")
    fields:         Optional[str]  = Field(default=None, description="Comma-separated fields to include")


@mcp.tool(
    name="shopify_list_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_products(params: ListProductsInput) -> str:
    """List products from the Shopify store with optional filters."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["status", "product_type", "vendor", "collection_id", "since_id", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data     = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="The Shopify product ID")


@mcp.tool(
    name="shopify_get_product",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_product(params: GetProductInput) -> str:
    """Retrieve a single product by ID, including all variants and images."""
    try:
        data = await _request("GET", f"products/{params.product_id}.json")
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class CreateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:        str                        = Field(..., min_length=1, description="Product title")
    body_html:    Optional[str]              = Field(default=None, description="HTML description")
    vendor:       Optional[str]              = Field(default=None)
    product_type: Optional[str]              = Field(default=None)
    tags:         Optional[str]              = Field(default=None, description="Comma-separated tags")
    status:       Optional[str]              = Field(default="draft", description="active, archived, or draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None, description="Variant objects with price, sku, etc.")
    options:      Optional[List[Dict[str, Any]]] = Field(default=None, description="Product options (Size, Color, etc.)")
    images:       Optional[List[Dict[str, Any]]] = Field(default=None, description="Image objects with src URL")


@mcp.tool(
    name="shopify_create_product",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_product(params: CreateProductInput) -> str:
    """Create a new product in the Shopify store."""
    try:
        product: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "vendor", "product_type", "tags", "status", "variants", "options", "images"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("POST", "products.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class UpdateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:   int            = Field(..., description="Product ID to update")
    title:        Optional[str]  = Field(default=None)
    body_html:    Optional[str]  = Field(default=None)
    vendor:       Optional[str]  = Field(default=None)
    product_type: Optional[str]  = Field(default=None)
    tags:         Optional[str]  = Field(default=None)
    status:       Optional[str]  = Field(default=None, description="active, archived, or draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None)


@mcp.tool(
    name="shopify_update_product",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_product(params: UpdateProductInput) -> str:
    """Update an existing product. Only provided fields are changed."""
    try:
        product: Dict[str, Any] = {}
        for field in ["title", "body_html", "vendor", "product_type", "tags", "status", "variants"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("PUT", f"products/{params.product_id}.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class DeleteProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID to delete")


@mcp.tool(
    name="shopify_delete_product",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_product(params: DeleteProductInput) -> str:
    """Permanently delete a product. This cannot be undone."""
    try:
        await _request("DELETE", f"products/{params.product_id}.json")
        return f"Product {params.product_id} deleted."
    except Exception as e:
        return _error(e)


class ProductCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:       Optional[str] = Field(default=None, description="active, archived, or draft")
    vendor:       Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_count_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_count_products(params: ProductCountInput) -> str:
    """Get the total count of products, optionally filtered."""
    try:
        p: Dict[str, Any] = {}
        for field in ["status", "vendor", "product_type"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# ORDERS
# ═══════════════════════════════════════════════════════════════════════════

class ListOrdersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:               Optional[int] = Field(default=50, ge=1, le=250)
    status:              Optional[str] = Field(default="any", description="open, closed, cancelled, any")
    financial_status:    Optional[str] = Field(default=None, description="authorized, pending, paid, refunded, voided, any")
    fulfillment_status:  Optional[str] = Field(default=None, description="shipped, partial, unshipped, unfulfilled, any")
    since_id:            Optional[int] = Field(default=None)
    created_at_min:      Optional[str] = Field(default=None, description="ISO 8601 date, e.g. 2024-01-01T00:00:00Z")
    created_at_max:      Optional[str] = Field(default=None)
    fields:              Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_orders(params: ListOrdersInput) -> str:
    """List orders with optional filters for status, financial/fulfillment status, and date range."""
    try:
        p: Dict[str, Any] = {"limit": params.limit, "status": params.status}
        for field in ["financial_status", "fulfillment_status", "since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data   = await _request("GET", "orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="The Shopify order ID")


@mcp.tool(
    name="shopify_get_order",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_order(params: GetOrderInput) -> str:
    """Retrieve a single order by ID with full details."""
    try:
        data = await _request("GET", f"orders/{params.order_id}.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


class OrderCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:             Optional[str] = Field(default="any")
    financial_status:   Optional[str] = Field(default=None)
    fulfillment_status: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_count_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_count_orders(params: OrderCountInput) -> str:
    """Get total order count, optionally filtered."""
    try:
        p: Dict[str, Any] = {"status": params.status}
        for field in ["financial_status", "fulfillment_status"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


class CloseOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="Order ID to close")


@mcp.tool(
    name="shopify_close_order",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_close_order(params: CloseOrderInput) -> str:
    """Close an order (marks it as completed)."""
    try:
        data = await _request("POST", f"orders/{params.order_id}/close.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


class CancelOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int            = Field(..., description="Order ID to cancel")
    reason:   Optional[str]  = Field(default=None, description="customer, fraud, inventory, declined, other")
    email:    Optional[bool] = Field(default=True,  description="Send cancellation email to customer")
    restock:  Optional[bool] = Field(default=False, description="Restock line items")


@mcp.tool(
    name="shopify_cancel_order",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_cancel_order(params: CancelOrderInput) -> str:
    """Cancel an order. Optionally restock items and notify the customer."""
    try:
        body: Dict[str, Any] = {}
        for field in ["reason", "email", "restock"]:
            val = getattr(params, field)
            if val is not None:
                body[field] = val
        data = await _request("POST", f"orders/{params.order_id}/cancel.json", body=body)
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════

class ListCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int] = Field(default=50, ge=1, le=250)
    since_id:       Optional[int] = Field(default=None)
    created_at_min: Optional[str] = Field(default=None, description="ISO 8601 date")
    created_at_max: Optional[str] = Field(default=None)
    fields:         Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_customers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_customers(params: ListCustomersInput) -> str:
    """List customers from the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for f in ["since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, f)
            if val is not None:
                p[f] = val
        data      = await _request("GET", "customers.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)


class SearchCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str           = Field(..., min_length=1, description="Search query (name, email, etc.)")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_search_customers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_search_customers(params: SearchCustomersInput) -> str:
    """Search customers by name, email, or other fields."""
    try:
        p         = {"query": params.query, "limit": params.limit}
        data      = await _request("GET", "customers/search.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)


class GetCustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(..., description="Shopify customer ID")


@mcp.tool(
    name="shopify_get_customer",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_customer(params: GetCustomerInput) -> str:
    """Retrieve a single customer by ID."""
    try:
        data = await _request("GET", f"customers/{params.customer_id}.json")
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class CreateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    first_name:         Optional[str]  = Field(default=None)
    last_name:          Optional[str]  = Field(default=None)
    email:              Optional[str]  = Field(default=None)
    phone:              Optional[str]  = Field(default=None)
    tags:               Optional[str]  = Field(default=None)
    note:               Optional[str]  = Field(default=None)
    addresses:          Optional[List[Dict[str, Any]]] = Field(default=None)
    send_email_invite:  Optional[bool] = Field(default=False)


@mcp.tool(
    name="shopify_create_customer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_customer(params: CreateCustomerInput) -> str:
    """Create a new customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note", "addresses", "send_email_invite"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("POST", "customers.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class UpdateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    customer_id: int           = Field(..., description="Customer ID to update")
    first_name:  Optional[str] = Field(default=None)
    last_name:   Optional[str] = Field(default=None)
    email:       Optional[str] = Field(default=None)
    phone:       Optional[str] = Field(default=None)
    tags:        Optional[str] = Field(default=None)
    note:        Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_customer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_customer(params: UpdateCustomerInput) -> str:
    """Update an existing customer. Only provided fields are changed."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("PUT", f"customers/{params.customer_id}.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class CustomerOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int           = Field(..., description="Customer ID")
    limit:       Optional[int] = Field(default=50, ge=1, le=250)
    status:      Optional[str] = Field(default="any")


@mcp.tool(
    name="shopify_get_customer_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_customer_orders(params: CustomerOrdersInput) -> str:
    """Get all orders for a specific customer."""
    try:
        p      = {"limit": params.limit, "status": params.status}
        data   = await _request("GET", f"customers/{params.customer_id}/orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS (Custom + Smart)
# ═══════════════════════════════════════════════════════════════════════════

class ListCollectionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:           Optional[int] = Field(default=50, ge=1, le=250)
    since_id:        Optional[int] = Field(default=None)
    collection_type: Optional[str] = Field(default="custom", description="'custom' or 'smart'")


@mcp.tool(
    name="shopify_list_collections",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_collections(params: ListCollectionsInput) -> str:
    """List custom or smart collections."""
    try:
        endpoint = "custom_collections.json" if params.collection_type == "custom" else "smart_collections.json"
        p: Dict[str, Any] = {"limit": params.limit}
        if params.since_id:
            p["since_id"] = params.since_id
        data = await _request("GET", endpoint, params=p)
        key  = "custom_collections" if params.collection_type == "custom" else "smart_collections"
        collections = data.get(key, [])
        return _fmt({"count": len(collections), "collections": collections})
    except Exception as e:
        return _error(e)


class GetCollectionProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: int           = Field(..., description="Collection ID")
    limit:         Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_get_collection_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_collection_products(params: GetCollectionProductsInput) -> str:
    """Get all products in a specific collection."""
    try:
        p        = {"limit": params.limit, "collection_id": params.collection_id}
        data     = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

class ListInventoryLocationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_list_locations",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_locations(params: ListInventoryLocationsInput) -> str:
    """List all inventory locations for the store."""
    try:
        data      = await _request("GET", "locations.json")
        locations = data.get("locations", [])
        return _fmt({"count": len(locations), "locations": locations})
    except Exception as e:
        return _error(e)


class GetInventoryLevelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location_id:         Optional[int] = Field(default=None, description="Filter by location ID")
    inventory_item_ids:  Optional[str] = Field(default=None, description="Comma-separated inventory item IDs")


@mcp.tool(
    name="shopify_get_inventory_levels",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_inventory_levels(params: GetInventoryLevelsInput) -> str:
    """Get inventory levels for specific locations or inventory items."""
    try:
        p: Dict[str, Any] = {}
        if params.location_id:
            p["location_ids"] = params.location_id
        if params.inventory_item_ids:
            p["inventory_item_ids"] = params.inventory_item_ids
        data   = await _request("GET", "inventory_levels.json", params=p)
        levels = data.get("inventory_levels", [])
        return _fmt({"count": len(levels), "inventory_levels": levels})
    except Exception as e:
        return _error(e)


class SetInventoryLevelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inventory_item_id: int = Field(..., description="Inventory item ID")
    location_id:       int = Field(..., description="Location ID")
    available:         int = Field(..., description="Available quantity to set")


@mcp.tool(
    name="shopify_set_inventory_level",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_inventory_level(params: SetInventoryLevelInput) -> str:
    """Set the available inventory for an item at a location."""
    try:
        body = {
            "inventory_item_id": params.inventory_item_id,
            "location_id":       params.location_id,
            "available":         params.available,
        }
        data = await _request("POST", "inventory_levels/set.json", body=body)
        return _fmt(data.get("inventory_level", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# FULFILLMENTS
# ═══════════════════════════════════════════════════════════════════════════

class ListFulfillmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int           = Field(..., description="Order ID")
    limit:    Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_fulfillments",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_fulfillments(params: ListFulfillmentsInput) -> str:
    """List fulfillments for a specific order."""
    try:
        p            = {"limit": params.limit}
        data         = await _request("GET", f"orders/{params.order_id}/fulfillments.json", params=p)
        fulfillments = data.get("fulfillments", [])
        return _fmt({"count": len(fulfillments), "fulfillments": fulfillments})
    except Exception as e:
        return _error(e)


class CreateFulfillmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id:         int                        = Field(..., description="Order ID to fulfill")
    location_id:      int                        = Field(..., description="Location ID fulfilling from")
    tracking_number:  Optional[str]              = Field(default=None)
    tracking_company: Optional[str]              = Field(default=None, description="e.g. UPS, FedEx, USPS")
    tracking_url:     Optional[str]              = Field(default=None)
    line_items:       Optional[List[Dict[str, Any]]] = Field(default=None, description="Specific line items (omit for all)")
    notify_customer:  Optional[bool]             = Field(default=True, description="Send shipping notification email")


@mcp.tool(
    name="shopify_create_fulfillment",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_fulfillment(params: CreateFulfillmentInput) -> str:
    """Create a fulfillment for an order (ship items)."""
    try:
        fulfillment: Dict[str, Any] = {"location_id": params.location_id}
        for field in ["tracking_number", "tracking_company", "tracking_url", "line_items", "notify_customer"]:
            val = getattr(params, field)
            if val is not None:
                fulfillment[field] = val
        data = await _request(
            "POST",
            f"orders/{params.order_id}/fulfillments.json",
            body={"fulfillment": fulfillment},
        )
        return _fmt(data.get("fulfillment", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SHOP INFO
# ═══════════════════════════════════════════════════════════════════════════

class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_get_shop",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_shop(params: EmptyInput) -> str:
    """Get store information: name, domain, plan, currency, timezone, etc."""
    try:
        data = await _request("GET", "shop.json")
        return _fmt(data.get("shop", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════

class ListWebhooksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    topic: Optional[str] = Field(default=None, description="Filter by topic, e.g. orders/create")


@mcp.tool(
    name="shopify_list_webhooks",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_webhooks(params: ListWebhooksInput) -> str:
    """List configured webhooks."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.topic:
            p["topic"] = params.topic
        data     = await _request("GET", "webhooks.json", params=p)
        webhooks = data.get("webhooks", [])
        return _fmt({"count": len(webhooks), "webhooks": webhooks})
    except Exception as e:
        return _error(e)


class CreateWebhookInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    topic:   str           = Field(..., description="Webhook topic, e.g. orders/create, products/update")
    address: str           = Field(..., description="URL to receive the webhook POST")
    format:  Optional[str] = Field(default="json", description="json or xml")


@mcp.tool(
    name="shopify_create_webhook",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_webhook(params: CreateWebhookInput) -> str:
    """Create a new webhook subscription."""
    try:
        webhook = {"topic": params.topic, "address": params.address, "format": params.format}
        data    = await _request("POST", "webhooks.json", body={"webhook": webhook})
        return _fmt(data.get("webhook", data))
    except Exception as e:
        return _error(e)

# ═══════════════════════════════════════════════════════════════════════════
# THEMES — Add this block to server.py BEFORE the final `mcp.run(...)` line
# ═══════════════════════════════════════════════════════════════════════════

class ListThemesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(
    name="shopify_list_themes",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_themes(params: ListThemesInput) -> str:
    """List all themes installed on the store. Returns theme ID, name, role (main/unpublished/demo), and timestamps."""
    try:
        data = await _request("GET", "themes.json")
        themes = data.get("themes", [])
        return _fmt({"count": len(themes), "themes": themes})
    except Exception as e:
        return _error(e)


class GetThemeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="The Shopify theme ID")

@mcp.tool(
    name="shopify_get_theme",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_theme(params: GetThemeInput) -> str:
    """Get details of a specific theme by ID."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}.json")
        return _fmt(data.get("theme", data))
    except Exception as e:
        return _error(e)


class ListThemeAssetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="The Shopify theme ID")

@mcp.tool(
    name="shopify_list_theme_assets",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_theme_assets(params: ListThemeAssetsInput) -> str:
    """List all file assets (Liquid templates, CSS, JS, images, etc.) in a theme. Returns asset keys (file paths)."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json")
        assets = data.get("assets", [])
        return _fmt({"count": len(assets), "assets": assets})
    except Exception as e:
        return _error(e)


class GetThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id: int = Field(..., description="The Shopify theme ID")
    asset_key: str = Field(..., description="Asset key/path, e.g. 'sections/header.liquid' or 'assets/custom.css'")

@mcp.tool(
    name="shopify_get_theme_asset",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_theme_asset(params: GetThemeAssetInput) -> str:
    """Read the content of a specific theme file (Liquid, CSS, JS, JSON, etc.) by its asset key."""
    try:
        data = await _request(
            "GET",
            f"themes/{params.theme_id}/assets.json",
            params={"asset[key]": params.asset_key},
        )
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)


class UpdateThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id: int = Field(..., description="The Shopify theme ID")
    asset_key: str = Field(..., description="Asset key/path, e.g. 'sections/header.liquid'")
    value: str = Field(..., description="The new file content (Liquid, CSS, JS, JSON, etc.)")

@mcp.tool(
    name="shopify_update_theme_asset",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_theme_asset(params: UpdateThemeAssetInput) -> str:
    """Create or update a theme file. If the asset key exists, it will be overwritten. If it doesn't exist, a new file is created."""
    try:
        data = await _request(
            "PUT",
            f"themes/{params.theme_id}/assets.json",
            body={"asset": {"key": params.asset_key, "value": params.value}},
        )
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)


class DeleteThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id: int = Field(..., description="The Shopify theme ID")
    asset_key: str = Field(..., description="Asset key/path to delete, e.g. 'assets/old-style.css'")

@mcp.tool(
    name="shopify_delete_theme_asset",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_delete_theme_asset(params: DeleteThemeAssetInput) -> str:
    """Permanently delete a theme file. This cannot be undone."""
    try:
        await _request(
            "DELETE",
            f"themes/{params.theme_id}/assets.json",
            params={"asset[key]": params.asset_key},
        )
        return _fmt({"deleted": True, "key": params.asset_key})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# MEDIA (product images) + SEO
# ═══════════════════════════════════════════════════════════════════════════

async def _graphql(query: str, variables: Optional[dict] = None, _retried: bool = False) -> dict:
    """GraphQL helper — mirrors _request's 401-refresh behaviour.
    Used for staged uploads and product media, which have no REST equivalent.
    """
    if not SHOPIFY_STORE:
        raise RuntimeError("Missing SHOPIFY_STORE environment variable.")

    url     = f"https://{SHOPIFY_STORE}.myshopify.com/admin/api/{API_VERSION}/graphql.json"
    headers = await _headers()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers=headers,
            json={"query": query, "variables": variables or {}},
            timeout=60.0,
        )

        if resp.status_code == 401 and not _retried and token_manager._use_client_credentials:
            logger.warning("GraphQL got 401 — refreshing token and retrying...")
            await token_manager.force_refresh()
            return await _graphql(query, variables, _retried=True)

        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL error: {json.dumps(payload['errors'])[:500]}")
        return payload.get("data", {})


class StageImageUploadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    filename:  str           = Field(..., min_length=1, description="File name, e.g. 'yhl-swag-frente.jpg' (used for SEO)")
    mime_type: str           = Field(..., description="MIME type, e.g. 'image/jpeg', 'image/webp', 'image/png'")
    file_size: str           = Field(..., description="File size in bytes, as a string, e.g. '348201'")
    resource:  Optional[str] = Field(default="PRODUCT_IMAGE", description="Staged resource type: PRODUCT_IMAGE (default) or FILE")


@mcp.tool(
    name="shopify_stage_image_upload",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_stage_image_upload(params: StageImageUploadInput) -> str:
    """Step 1 of uploading a LOCAL image. Creates a staged upload target and returns a pre-signed
    'url' + 'parameters' to POST the raw file bytes to (done from the local machine — no token needed),
    plus a 'resourceUrl' to pass to shopify_attach_product_media afterwards."""
    try:
        query = (
            "mutation stage($input:[StagedUploadInput!]!){"
            " stagedUploadsCreate(input:$input){"
            " stagedTargets{ url resourceUrl parameters{ name value } }"
            " userErrors{ field message } } }"
        )
        variables = {"input": [{
            "filename":   params.filename,
            "mimeType":   params.mime_type,
            "resource":   params.resource,
            "fileSize":   params.file_size,
            "httpMethod": "POST",
        }]}
        data   = await _graphql(query, variables)
        result = data.get("stagedUploadsCreate", {})
        errs   = result.get("userErrors") or []
        if errs:
            return _fmt({"userErrors": errs})
        return _fmt({"stagedTargets": result.get("stagedTargets", [])})
    except Exception as e:
        return _error(e)


class AttachProductMediaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID to attach media to")
    sources: List[Dict[str, Any]] = Field(
        ...,
        description="List of media to attach. Each item: {original_source: <public image URL or staged resourceUrl>, alt: <optional alt text>}",
    )


@mcp.tool(
    name="shopify_attach_product_media",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_attach_product_media(params: AttachProductMediaInput) -> str:
    """Step 2: attach one or more images to a product. Each 'original_source' can be a public image URL
    OR a 'resourceUrl' returned by shopify_stage_image_upload. Sets alt text for SEO/accessibility."""
    try:
        media: List[Dict[str, Any]] = []
        for s in params.sources:
            src = s.get("original_source") or s.get("originalSource")
            if not src:
                return _fmt({"error": "each source needs 'original_source'"})
            item: Dict[str, Any] = {"mediaContentType": "IMAGE", "originalSource": src}
            if s.get("alt"):
                item["alt"] = s["alt"]
            media.append(item)

        query = (
            "mutation attach($productId:ID!,$media:[CreateMediaInput!]!){"
            " productCreateMedia(productId:$productId, media:$media){"
            " media{ ... on MediaImage { id alt status image{ url } } }"
            " mediaUserErrors{ field message } } }"
        )
        variables = {"productId": f"gid://shopify/Product/{params.product_id}", "media": media}
        data = await _graphql(query, variables)
        return _fmt(data.get("productCreateMedia", data))
    except Exception as e:
        return _error(e)


class SetProductSeoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:      int           = Field(..., description="Product ID")
    seo_title:       Optional[str] = Field(default=None, description="Meta title (global.title_tag) shown in search results")
    seo_description: Optional[str] = Field(default=None, description="Meta description (global.description_tag) shown in search results")


@mcp.tool(
    name="shopify_set_product_seo",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_product_seo(params: SetProductSeoInput) -> str:
    """Set a product's SEO meta title and/or description (the snippet shown on Google).
    Upserts the global.title_tag / global.description_tag metafields."""
    try:
        if params.seo_title is None and params.seo_description is None:
            return "Error: provide seo_title and/or seo_description."

        existing = await _request("GET", f"products/{params.product_id}/metafields.json", params={"namespace": "global"})
        metas    = {m["key"]: m for m in existing.get("metafields", [])}

        targets = []
        if params.seo_title is not None:
            targets.append(("title_tag", params.seo_title, "single_line_text_field"))
        if params.seo_description is not None:
            targets.append(("description_tag", params.seo_description, "multi_line_text_field"))

        results = []
        for key, value, mtype in targets:
            if key in metas:
                mid  = metas[key]["id"]
                data = await _request("PUT", f"metafields/{mid}.json",
                                      body={"metafield": {"id": mid, "value": value, "type": mtype}})
            else:
                data = await _request("POST", f"products/{params.product_id}/metafields.json",
                                      body={"metafield": {"namespace": "global", "key": key, "value": value, "type": mtype}})
            results.append(data.get("metafield", data))
        return _fmt({"updated": results})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# METAFIELDS · FILES · VARIANT MEDIA   (added jun 2026 — product setup, both stores)
# ═══════════════════════════════════════════════════════════════════════════

class SetProductMetafieldInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id: int = Field(..., description="Product ID that owns the metafield")
    namespace:  str = Field(..., description="Metafield namespace, e.g. 'custom'")
    key:        str = Field(..., description="Metafield key, e.g. 'guia_cuidados'")
    type:       str = Field(..., description="Metafield type: e.g. 'multi_line_text_field', 'single_line_text_field', 'file_reference'")
    value:      str = Field(..., description="Value. For file_reference pass the file GID (gid://shopify/MediaImage/...)")


@mcp.tool(
    name="shopify_set_product_metafield",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_product_metafield(params: SetProductMetafieldInput) -> str:
    """Upsert ANY product metafield (namespace/key/type/value) via GraphQL metafieldsSet.
    E.g. custom.guia_cuidados (multi_line_text_field) or custom.guia_tallas (file_reference → file GID)."""
    try:
        query = (
            "mutation setMeta($metafields:[MetafieldsSetInput!]!){"
            " metafieldsSet(metafields:$metafields){"
            " metafields{ id namespace key type value }"
            " userErrors{ field message } } }"
        )
        variables = {"metafields": [{
            "ownerId":   f"gid://shopify/Product/{params.product_id}",
            "namespace": params.namespace,
            "key":       params.key,
            "type":      params.type,
            "value":     params.value,
        }]}
        data = await _graphql(query, variables)
        return _fmt(data.get("metafieldsSet", data))
    except Exception as e:
        return _error(e)


class CreateFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    original_source: str           = Field(..., description="Public image URL or a staged resourceUrl (resource=FILE) to import into Shopify Files")
    alt:             Optional[str] = Field(default=None, description="Alt text for the file")
    content_type:    Optional[str] = Field(default="IMAGE", description="File content type: IMAGE (default), FILE or VIDEO")


@mcp.tool(
    name="shopify_create_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_file(params: CreateFileInput) -> str:
    """Import a file into Shopify *Files* (fileCreate) and return its GID — use that GID as the value of a
    file_reference metafield (e.g. custom.guia_tallas). For local files: stage with resource=FILE, POST the
    bytes, then pass the resourceUrl here."""
    try:
        f: Dict[str, Any] = {"originalSource": params.original_source, "contentType": params.content_type or "IMAGE"}
        if params.alt:
            f["alt"] = params.alt
        query = (
            "mutation createFile($files:[FileCreateInput!]!){"
            " fileCreate(files:$files){"
            " files{ id fileStatus alt ... on MediaImage { image{ url } } }"
            " userErrors{ field message } } }"
        )
        data = await _graphql(query, {"files": [f]})
        return _fmt(data.get("fileCreate", data))
    except Exception as e:
        return _error(e)


class SetVariantImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id: int = Field(..., description="Product ID that owns the variant and the media")
    variant_id: int = Field(..., description="Variant ID to set the image on")
    media_id:   str = Field(..., description="Media GID to assign (gid://shopify/MediaImage/...), already attached to the product")


@mcp.tool(
    name="shopify_set_variant_image",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_set_variant_image(params: SetVariantImageInput) -> str:
    """Assign a product media image to a specific variant (productVariantAppendMedia). The media must already
    be attached to the product (shopify_attach_product_media returns the media ids you pass here)."""
    try:
        query = (
            "mutation appendMedia($productId:ID!,$variantMedia:[ProductVariantAppendMediaInput!]!){"
            " productVariantAppendMedia(productId:$productId, variantMedia:$variantMedia){"
            " productVariants{ id } userErrors{ field message } } }"
        )
        variables = {
            "productId": f"gid://shopify/Product/{params.product_id}",
            "variantMedia": [{
                "variantId": f"gid://shopify/ProductVariant/{params.variant_id}",
                "mediaIds":  [params.media_id],
            }],
        }
        data = await _graphql(query, variables)
        return _fmt(data.get("productVariantAppendMedia", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# RAW GRAPHQL (read-only escape hatch — added jul 2026, for schema introspection
# and anything without a dedicated tool yet)
# ═══════════════════════════════════════════════════════════════════════════

class GraphqlQueryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query:     str            = Field(..., description="A GraphQL QUERY (read-only) — introspection or any `query { ... }`")
    variables: Optional[dict] = Field(default=None, description="Optional GraphQL variables")


@mcp.tool(
    name="shopify_graphql_query",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_graphql_query(params: GraphqlQueryInput) -> str:
    """Run an arbitrary READ-ONLY GraphQL query against the Admin API (e.g. `{ __type(name:"Foo"){ name fields { name } } }`).
    Use this to check exact field/input names before adding a dedicated tool, or for reads with no tool yet.
    Does not accept mutations — use a dedicated write tool, or ask for one to be added."""
    try:
        if "mutation" in params.query.lower().split("(")[0].split("{")[0]:
            return _error(RuntimeError("shopify_graphql_query is read-only — mutations aren't allowed here."))
        data = await _graphql(params.query, params.variables)
        return _fmt(data)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# METAFIELD DEFINITIONS (storefront filters — added jul 2026)
# ═══════════════════════════════════════════════════════════════════════════

class GetMetafieldDefinitionsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    owner_type: Optional[str] = Field(default="PRODUCT", description="Owner type: PRODUCT, VARIANT, COLLECTION, etc.")
    namespace:  Optional[str] = Field(default=None, description="Filter by namespace, e.g. 'custom'")


@mcp.tool(
    name="shopify_get_metafield_definitions",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_metafield_definitions(params: GetMetafieldDefinitionsInput) -> str:
    """List metafield DEFINITIONS for an owner type (default PRODUCT), optionally filtered by namespace.
    Shows id/name/key/type and whether each already has the 'filterable' capability (storefront filter)
    enabled. Check this before shopify_set_metafield_definition to see what already exists."""
    try:
        query = (
            "query defs($ownerType:MetafieldOwnerType!,$namespace:String){"
            " metafieldDefinitions(first:100, ownerType:$ownerType, namespace:$namespace){"
            " nodes{ id name namespace key type{ name } "
            " validations{ name value } capabilities{ filterable{ enabled } } } } }"
        )
        variables = {"ownerType": params.owner_type, "namespace": params.namespace}
        data = await _graphql(query, variables)
        return _fmt(data.get("metafieldDefinitions", data))
    except Exception as e:
        return _error(e)


class SetMetafieldDefinitionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    namespace:  str                 = Field(..., description="Metafield namespace, e.g. 'custom'")
    key:        str                 = Field(..., description="Metafield key, e.g. 'genero'")
    name:       str                 = Field(..., description="Human-readable name shown in admin, e.g. 'Género'")
    type:       str                 = Field(default="single_line_text_field", description="Metafield type, e.g. 'single_line_text_field'")
    owner_type: Optional[str]       = Field(default="PRODUCT", description="Owner type: PRODUCT, VARIANT, COLLECTION, etc.")
    choices:    Optional[List[str]] = Field(default=None, description="Optional fixed list of allowed values (dropdown, keeps filter values consistent), e.g. ['Hombre','Mujer','Unisex']")


@mcp.tool(
    name="shopify_set_metafield_definition",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_metafield_definition(params: SetMetafieldDefinitionInput) -> str:
    """Create (or update) a metafield DEFINITION with the 'filterable' capability enabled, so it can show up
    as a storefront filter in Search & Discovery. NOTE: Shopify still requires a one-time manual toggle in
    Admin > Settings > Search & discovery > Filters to switch it visibly ON for shoppers the first time —
    this tool only prepares the definition (create-or-update, idempotent). Pass `choices` for a fixed
    dropdown of values (recommended for filters, e.g. Género: Hombre/Mujer/Unisex)."""
    try:
        validations = []
        if params.choices:
            validations.append({"name": "choices", "value": json.dumps(params.choices)})

        definition_input = {
            "name": params.name,
            "namespace": params.namespace,
            "key": params.key,
            "type": params.type,
            "ownerType": params.owner_type,
            "capabilities": {"filterable": {"enabled": True}},
        }
        if validations:
            definition_input["validations"] = validations

        create_query = (
            "mutation createDef($definition:MetafieldDefinitionInput!){"
            " metafieldDefinitionCreate(definition:$definition){"
            " createdDefinition{ id name } userErrors{ field message code } } }"
        )
        data = await _graphql(create_query, {"definition": definition_input})
        result = data.get("metafieldDefinitionCreate", {})
        errors = result.get("userErrors") or []

        if not any(e.get("code") == "TAKEN" for e in errors):
            return _fmt(result)

        # Ya existe -> buscar su id y activarle/actualizarle la capacidad filterable
        lookup_query = (
            "query defs($ownerType:MetafieldOwnerType!,$namespace:String,$key:String!){"
            " metafieldDefinitions(first:1, ownerType:$ownerType, namespace:$namespace, key:$key){"
            " nodes{ id } } }"
        )
        lookup = await _graphql(lookup_query, {
            "ownerType": params.owner_type, "namespace": params.namespace, "key": params.key,
        })
        nodes = (lookup.get("metafieldDefinitions") or {}).get("nodes") or []
        if not nodes:
            return _fmt({"createErrors": errors})

        update_input = {
            "id": nodes[0]["id"],
            "name": params.name,
            "capabilities": {"filterable": {"enabled": True}},
        }
        if validations:
            update_input["validations"] = validations

        update_query = (
            "mutation updateDef($definition:MetafieldDefinitionUpdateInput!){"
            " metafieldDefinitionUpdate(definition:$definition){"
            " updatedDefinition{ id name } userErrors{ field message code } } }"
        )
        update_data = await _graphql(update_query, {"definition": update_input})
        return _fmt(update_data.get("metafieldDefinitionUpdate", update_data))
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
