# Shopping Cart API Documentation

## Overview

The Shopping Cart API provides programmatic access to shopping cart functionality, enabling developers to integrate cart operations into applications and services. The API supports all cart operations including adding items, removing items, viewing contents, and managing cart state.

## Base URL

```
https://api.example.com/v1
```

## Authentication

The API uses session-based authentication. All requests must include a valid session ID either as a path parameter or in the request headers.

### Session Management

```http
# Session ID in path (recommended)
GET /api/cart/{session_id}

# Session ID in header (alternative)
GET /api/cart
X-Session-ID: your-session-id
```

## API Endpoints

### 1. Get Cart Contents

Retrieve all items in a user's shopping cart.

```http
GET /api/cart/{session_id}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | Unique session identifier |

#### Response

```json
{
  "success": true,
  "message": "Cart retrieved successfully",
  "cart_data": {
    "session_id": "sess_123456789",
    "items": [
      {
        "id": "item_001",
        "product_id": "prod_laptop_001",
        "product_title": "ASUS Gaming Laptop",
        "quantity": 1,
        "price": 899.99,
        "product_metadata": {
          "brand": "ASUS",
          "category": "Laptops",
          "image_url": "https://example.com/laptop.jpg"
        },
        "added_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:30:00Z"
      }
    ],
    "total_items": 1,
    "total_quantity": 1,
    "total_value": 899.99,
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

#### Error Responses

```json
{
  "success": false,
  "error": "Session not found",
  "error_code": "SESSION_NOT_FOUND",
  "message": "The specified session ID does not exist"
}
```

### 2. Add Item to Cart

Add a product to the shopping cart or update quantity if item already exists.

```http
POST /api/cart/{session_id}/add
Content-Type: application/json
```

#### Request Body

```json
{
  "product_id": "prod_laptop_001",
  "product_title": "ASUS Gaming Laptop",
  "quantity": 1,
  "price": 899.99,
  "product_metadata": {
    "brand": "ASUS",
    "category": "Laptops",
    "image_url": "https://example.com/laptop.jpg",
    "specifications": {
      "ram": "16GB",
      "storage": "512GB SSD",
      "processor": "Intel i7"
    }
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| product_id | string | Yes | Unique product identifier |
| product_title | string | Yes | Human-readable product name |
| quantity | integer | No | Number of items (default: 1) |
| price | number | No | Product price |
| product_metadata | object | No | Additional product information |

#### Response

```json
{
  "success": true,
  "message": "Item added to cart successfully",
  "operation": "add",
  "cart_data": {
    "session_id": "sess_123456789",
    "items": [...],
    "total_items": 2,
    "total_quantity": 3,
    "total_value": 1299.98,
    "last_updated": "2024-01-15T10:35:00Z"
  },
  "added_item": {
    "id": "item_002",
    "product_id": "prod_laptop_001",
    "product_title": "ASUS Gaming Laptop",
    "quantity": 1,
    "price": 899.99,
    "added_at": "2024-01-15T10:35:00Z"
  }
}
```

#### Error Responses

```json
{
  "success": false,
  "error": "Invalid quantity",
  "error_code": "INVALID_QUANTITY",
  "message": "Quantity must be a positive integer",
  "details": {
    "provided_quantity": -1,
    "valid_range": "1 to 999"
  }
}
```

### 3. Remove Item from Cart

Remove a specific item from the cart or reduce its quantity.

```http
DELETE /api/cart/{session_id}/remove/{product_id}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | Unique session identifier |
| product_id | string | Yes | Product identifier to remove |
| quantity | integer | No | Quantity to remove (query param) |

#### Query Parameters

```http
DELETE /api/cart/{session_id}/remove/{product_id}?quantity=2
```

#### Response

```json
{
  "success": true,
  "message": "Item removed from cart successfully",
  "operation": "remove",
  "cart_data": {
    "session_id": "sess_123456789",
    "items": [...],
    "total_items": 1,
    "total_quantity": 1,
    "total_value": 399.99,
    "last_updated": "2024-01-15T10:40:00Z"
  },
  "removed_item": {
    "product_id": "prod_laptop_001",
    "product_title": "ASUS Gaming Laptop",
    "quantity_removed": 1,
    "remaining_quantity": 0
  }
}
```

### 4. Update Item Quantity

Update the quantity of a specific item in the cart.

```http
PUT /api/cart/{session_id}/update/{product_id}
Content-Type: application/json
```

#### Request Body

```json
{
  "quantity": 3
}
```

#### Response

```json
{
  "success": true,
  "message": "Item quantity updated successfully",
  "operation": "update",
  "cart_data": {
    "session_id": "sess_123456789",
    "items": [...],
    "total_items": 1,
    "total_quantity": 3,
    "total_value": 2699.97,
    "last_updated": "2024-01-15T10:45:00Z"
  },
  "updated_item": {
    "product_id": "prod_laptop_001",
    "product_title": "ASUS Gaming Laptop",
    "old_quantity": 1,
    "new_quantity": 3
  }
}
```

### 5. Clear Cart

Remove all items from the shopping cart.

```http
DELETE /api/cart/{session_id}/clear
```

#### Response

```json
{
  "success": true,
  "message": "Cart cleared successfully",
  "operation": "clear",
  "cart_data": {
    "session_id": "sess_123456789",
    "items": [],
    "total_items": 0,
    "total_quantity": 0,
    "total_value": 0.00,
    "last_updated": "2024-01-15T10:50:00Z"
  },
  "cleared_items_count": 5
}
```

### 6. Get Cart Summary

Get a summary of cart contents without full item details.

```http
GET /api/cart/{session_id}/summary
```

#### Response

```json
{
  "success": true,
  "message": "Cart summary retrieved successfully",
  "cart_summary": {
    "session_id": "sess_123456789",
    "total_items": 3,
    "total_quantity": 7,
    "total_value": 1899.97,
    "last_updated": "2024-01-15T10:30:00Z",
    "categories": {
      "Laptops": 2,
      "Accessories": 1
    },
    "price_range": {
      "min": 29.99,
      "max": 899.99,
      "average": 271.42
    }
  }
}
```

## Enhanced Query Response Integration

### LangGraph API Integration

The shopping cart integrates with the main LangGraph API to provide enhanced responses that include cart data.

```http
POST /api/query
Content-Type: application/json
```

#### Request Body

```json
{
  "query": "Add this laptop to my cart",
  "session_id": "sess_123456789",
  "include_cart_data": true
}
```

#### Enhanced Response

```json
{
  "query": "Add this laptop to my cart",
  "response": "I've added the ASUS Gaming Laptop to your cart. Your cart now contains 2 items.",
  "session_id": "sess_123456789",
  "conversation_turn": 5,
  "agent_used": "shopping_cart_agent",
  "routing_decision": "cart",
  "tools_called": ["add_to_cart"],
  "cart_data": {
    "items": [...],
    "total_items": 2,
    "total_quantity": 3,
    "total_value": 1299.98,
    "last_updated": "2024-01-15T10:35:00Z"
  },
  "cart_updated": true,
  "cart_item_count": 2,
  "cart_total": 1299.98
}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "success": false,
  "error": "Error description",
  "error_code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    "additional": "error details"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Common Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| SESSION_NOT_FOUND | 404 | Session ID does not exist |
| PRODUCT_NOT_FOUND | 404 | Product ID not found in cart |
| INVALID_QUANTITY | 400 | Invalid quantity value |
| CART_OPERATION_FAILED | 500 | Database operation failed |
| VALIDATION_ERROR | 400 | Request validation failed |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Unexpected server error |

### Error Recovery

The API implements automatic retry logic for transient errors:

- **Database Connection Errors**: Automatic retry with exponential backoff
- **Timeout Errors**: Retry up to 3 times
- **Validation Errors**: Return detailed validation messages

## Rate Limiting

### Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| GET /api/cart/* | 100 requests | 1 minute |
| POST /api/cart/*/add | 50 requests | 1 minute |
| DELETE /api/cart/*/remove/* | 50 requests | 1 minute |
| DELETE /api/cart/*/clear | 10 requests | 1 minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## Data Models

### Cart Item Model

```typescript
interface CartItem {
  id: string;                    // Unique item identifier
  product_id: string;           // Product identifier
  product_title: string;        // Product name
  quantity: number;             // Item quantity
  price?: number;               // Product price
  product_metadata?: {          // Additional product data
    brand?: string;
    category?: string;
    image_url?: string;
    specifications?: object;
    [key: string]: any;
  };
  added_at: string;             // ISO timestamp
  updated_at: string;           // ISO timestamp
}
```

### Cart Data Model

```typescript
interface CartData {
  session_id: string;           // Session identifier
  items: CartItem[];            // Array of cart items
  total_items: number;          // Unique product count
  total_quantity: number;       // Sum of all quantities
  total_value?: number;         // Sum of all prices
  last_updated: string;         // ISO timestamp
}
```

### API Response Model

```typescript
interface ApiResponse<T = any> {
  success: boolean;             // Operation success status
  message: string;              // Human-readable message
  data?: T;                     // Response data
  error?: string;               // Error description
  error_code?: string;          // Machine-readable error code
  details?: object;             // Additional error details
  timestamp?: string;           // Response timestamp
  request_id?: string;          // Request identifier
}
```

## SDK and Client Libraries

### JavaScript/TypeScript SDK

```typescript
import { ShoppingCartAPI } from '@company/shopping-cart-sdk';

const cartAPI = new ShoppingCartAPI({
  baseURL: 'https://api.example.com/v1',
  sessionId: 'sess_123456789'
});

// Add item to cart
const result = await cartAPI.addItem({
  product_id: 'prod_laptop_001',
  product_title: 'ASUS Gaming Laptop',
  quantity: 1,
  price: 899.99
});

// Get cart contents
const cart = await cartAPI.getCart();

// Remove item
await cartAPI.removeItem('prod_laptop_001');
```

### Python SDK

```python
from shopping_cart_sdk import ShoppingCartAPI

cart_api = ShoppingCartAPI(
    base_url='https://api.example.com/v1',
    session_id='sess_123456789'
)

# Add item to cart
result = cart_api.add_item(
    product_id='prod_laptop_001',
    product_title='ASUS Gaming Laptop',
    quantity=1,
    price=899.99
)

# Get cart contents
cart = cart_api.get_cart()

# Remove item
cart_api.remove_item('prod_laptop_001')
```

## Webhooks

### Cart Update Webhooks

Register webhooks to receive notifications when cart contents change:

```http
POST /api/webhooks/register
Content-Type: application/json
```

#### Request Body

```json
{
  "url": "https://your-app.com/webhooks/cart-update",
  "events": ["cart.item_added", "cart.item_removed", "cart.cleared"],
  "session_id": "sess_123456789"
}
```

#### Webhook Payload

```json
{
  "event": "cart.item_added",
  "session_id": "sess_123456789",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "item": {
      "product_id": "prod_laptop_001",
      "product_title": "ASUS Gaming Laptop",
      "quantity": 1
    },
    "cart_summary": {
      "total_items": 2,
      "total_quantity": 3
    }
  }
}
```

## Testing

### Test Environment

```
Base URL: https://api-test.example.com/v1
```

### Test Session IDs

Use these session IDs for testing:

- `test_session_empty`: Empty cart
- `test_session_populated`: Cart with sample items
- `test_session_error`: Triggers error responses

### Sample Test Requests

```bash
# Test adding item
curl -X POST https://api-test.example.com/v1/cart/test_session_empty/add \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "test_product_001",
    "product_title": "Test Product",
    "quantity": 1,
    "price": 99.99
  }'

# Test getting cart
curl https://api-test.example.com/v1/cart/test_session_populated

# Test removing item
curl -X DELETE https://api-test.example.com/v1/cart/test_session_populated/remove/test_product_001
```

## Performance Considerations

### Response Times

- **GET operations**: < 100ms average
- **POST/PUT operations**: < 200ms average
- **DELETE operations**: < 150ms average

### Caching

- Cart data is cached for 5 minutes
- Cache invalidation on cart modifications
- ETag support for conditional requests

### Pagination

For large carts (>100 items):

```http
GET /api/cart/{session_id}?page=1&limit=50
```

## Security

### Data Protection

- All API communications use HTTPS
- Session IDs are cryptographically secure
- No personal data stored in cart items
- Automatic session cleanup after inactivity

### Access Control

- Session-based isolation
- No cross-session data access
- Rate limiting to prevent abuse
- Input validation and sanitization

## Monitoring and Analytics

### Available Metrics

- API response times
- Error rates by endpoint
- Cart operation frequencies
- Session activity patterns

### Health Check

```http
GET /api/health/cart
```

#### Response

```json
{
  "status": "healthy",
  "database": "connected",
  "cache": "operational",
  "response_time": "45ms"
}
```

## Changelog

### Version 1.0.0 (Current)
- Initial API release
- Basic CRUD operations
- Session-based cart management
- Error handling and recovery

### Planned Features (v1.1.0)
- Bulk operations
- Cart sharing capabilities
- Advanced filtering and search
- Real-time updates via WebSocket

## Support

### Documentation
- API Reference: https://docs.example.com/api/cart
- SDK Documentation: https://docs.example.com/sdk
- Tutorials: https://docs.example.com/tutorials

### Contact
- Technical Support: api-support@example.com
- Bug Reports: https://github.com/company/shopping-cart-api/issues
- Feature Requests: https://github.com/company/shopping-cart-api/discussions