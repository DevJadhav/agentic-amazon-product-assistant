# Shopping Cart Functionality User Guide

## Overview

The Shopping Cart functionality allows users to manage a persistent shopping cart during their product research and comparison sessions. Users can add products to their cart, remove items, view cart contents, and manage quantities through natural language interactions with the AI assistant.

## Getting Started

### Accessing the Shopping Cart

The shopping cart is available through two interfaces:

1. **Chat Interface**: Use natural language commands to manage your cart
2. **Sidebar Tab**: View and manage your cart through the dedicated cart tab in the sidebar

### Basic Cart Operations

#### Adding Items to Cart

You can add products to your cart using natural language:

**Examples:**
- "Add this laptop to my cart"
- "Put the iPhone 13 in my basket"
- "I want to add 2 of these headphones to my cart"
- "Include the Samsung TV in my shopping cart"

**What happens:**
- The system identifies the product from your conversation context
- Adds the item with specified quantity (default: 1)
- Confirms the addition with product details
- Updates the cart display in real-time

#### Removing Items from Cart

Remove items using natural language commands:

**Examples:**
- "Remove the laptop from my cart"
- "Take out the iPhone from my basket"
- "Delete 1 headphone from my cart"
- "Remove all Samsung TVs from my cart"

**What happens:**
- The system identifies the product to remove
- Removes specified quantity or all items if no quantity specified
- Confirms the removal
- Updates the cart display

#### Viewing Cart Contents

Check your cart contents anytime:

**Examples:**
- "Show me my cart"
- "What's in my shopping cart?"
- "List my cart items"
- "Display my basket contents"

**What you'll see:**
- Complete list of cart items
- Quantities for each item
- Product details (name, price if available)
- Total item count
- Cart summary information

#### Clearing Your Cart

Remove all items at once:

**Examples:**
- "Clear my cart"
- "Empty my shopping cart"
- "Remove everything from my basket"

## Advanced Features

### Quantity Management

#### Adding Multiple Items
- "Add 3 of these wireless mice to my cart"
- "Put 2 laptops in my basket"

#### Updating Quantities
- "Change the laptop quantity to 2 in my cart"
- "Update the headphones to 4 pieces"

#### Partial Removal
- "Remove 1 laptop from my cart" (keeps remaining items)
- "Take out 2 headphones" (reduces quantity by 2)

### Product Context Integration

The cart system understands your conversation context:

- **After Product Search**: "Add this one to my cart" (refers to the last discussed product)
- **During Comparisons**: "Add the better laptop to my cart" (uses comparison context)
- **From Recommendations**: "Put the recommended phone in my basket"

### Session Persistence

Your cart persists across sessions:
- Items remain in your cart when you return
- Cart state is maintained across browser refreshes
- Session-based isolation ensures privacy

## Cart Display Interface

### Sidebar Cart Tab

The cart tab in the sidebar provides:

#### Cart Summary
- **Total Items**: Number of unique products
- **Total Quantity**: Sum of all item quantities
- **Cart Status**: Empty, has items, or error states

#### Item Details
For each cart item:
- **Product Name**: Full product title
- **Quantity**: Number of items
- **Product Details**: Additional product information
- **Add Date**: When item was added to cart

#### Empty Cart State
When your cart is empty:
- Friendly message indicating empty cart
- Suggestions for getting started
- Links to product search functionality

### Real-Time Updates

The cart display updates automatically:
- **Immediate Refresh**: Changes appear instantly after operations
- **Visual Feedback**: Loading states during operations
- **Error Handling**: Clear error messages if operations fail

## Error Handling and Troubleshooting

### Common Issues and Solutions

#### "Product not found in conversation"
**Problem**: System can't identify which product to add
**Solution**: Be more specific about the product name or refer to a recently discussed item

#### "Cart operation failed"
**Problem**: Database or system error
**Solution**: Try the operation again, or contact support if persistent

#### "Invalid quantity specified"
**Problem**: Quantity is zero, negative, or not a number
**Solution**: Specify a positive whole number for quantity

#### "Item not in cart"
**Problem**: Trying to remove an item that's not in your cart
**Solution**: Check your cart contents first, ensure correct product name

### Best Practices

#### For Successful Cart Operations

1. **Be Specific**: Use clear product names when adding items
2. **Check Context**: Ensure you've discussed the product recently
3. **Verify Quantities**: Double-check quantities before adding
4. **Review Cart**: Regularly check your cart contents

#### For Better Experience

1. **Use Natural Language**: The system understands conversational commands
2. **Combine Operations**: You can search and add in the same conversation
3. **Ask for Clarification**: If unsure, ask "What's in my cart?" first
4. **Use Sidebar**: Check the cart tab for visual confirmation

## Integration with Product Research

### Seamless Workflow

The cart integrates naturally with product research:

1. **Search Products**: "Find me gaming laptops under $1000"
2. **Compare Options**: "Compare the top 3 results"
3. **Add to Cart**: "Add the ASUS laptop to my cart"
4. **Continue Research**: "Now show me gaming mice"
5. **Add More Items**: "Add the Logitech mouse too"

### Context Awareness

The system maintains context across operations:
- Remembers recently discussed products
- Understands references like "this one", "the better option"
- Links product comparisons to cart additions

## Privacy and Data Management

### Session Isolation
- Your cart is private to your session
- Other users cannot see or modify your cart
- Cart data is isolated by session ID

### Data Persistence
- Cart data is stored securely in the database
- Items persist across browser sessions
- Automatic cleanup of old cart data

### Data Security
- No personal information required for cart functionality
- Session-based identification only
- Secure database storage with proper access controls

## API Integration

### For Developers

The cart functionality is available through API endpoints:

#### Get Cart Contents
```http
GET /api/cart/{session_id}
```

#### Add Item to Cart
```http
POST /api/cart/{session_id}/add
Content-Type: application/json

{
  "product_id": "string",
  "product_title": "string",
  "quantity": 1,
  "price": 99.99,
  "metadata": {}
}
```

#### Remove Item from Cart
```http
DELETE /api/cart/{session_id}/remove/{product_id}
```

#### Clear Cart
```http
DELETE /api/cart/{session_id}/clear
```

### Response Format

All cart operations return standardized responses:

```json
{
  "success": true,
  "message": "Item added to cart successfully",
  "cart_data": {
    "items": [...],
    "total_items": 3,
    "total_quantity": 5,
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

## Frequently Asked Questions

### General Questions

**Q: How long do items stay in my cart?**
A: Items persist across sessions and are automatically cleaned up after extended periods of inactivity.

**Q: Can I share my cart with others?**
A: No, carts are session-private and cannot be shared directly.

**Q: Is there a limit to cart size?**
A: There's no hard limit, but very large carts may impact performance.

### Technical Questions

**Q: What happens if I use multiple browser tabs?**
A: All tabs share the same session cart, so changes appear across all tabs.

**Q: Does the cart work offline?**
A: No, cart operations require an active internet connection.

**Q: Can I export my cart data?**
A: Currently, cart data is view-only through the interface.

### Troubleshooting

**Q: My cart appears empty after returning**
A: Check if you're using the same browser and haven't cleared cookies.

**Q: Cart operations are slow**
A: This may indicate high system load; operations should complete within a few seconds normally.

**Q: I can't add a specific product**
A: Ensure the product was discussed in your recent conversation or try being more specific about the product name.

## Support and Feedback

### Getting Help

If you encounter issues with cart functionality:

1. **Check Error Messages**: Read any error messages carefully
2. **Try Again**: Many issues are transient and resolve on retry
3. **Clear Context**: Start a new conversation if context seems confused
4. **Check Sidebar**: Use the cart tab to verify current cart state

### Providing Feedback

Help us improve the cart functionality:
- Report bugs or unexpected behavior
- Suggest improvements to the user experience
- Share use cases we haven't considered

## Future Enhancements

### Planned Features

- **Cart Sharing**: Share cart contents with others
- **Save for Later**: Move items to a wishlist
- **Price Tracking**: Monitor price changes for cart items
- **Bulk Operations**: Add multiple items at once
- **Cart Templates**: Save and reuse common cart configurations

### Integration Improvements

- **Checkout Integration**: Direct integration with e-commerce platforms
- **Inventory Checking**: Real-time availability verification
- **Price Comparison**: Automatic price comparison across retailers
- **Recommendation Engine**: Suggest related items for cart contents

## Conclusion

The Shopping Cart functionality provides a seamless way to manage your product selections during research and comparison activities. By understanding natural language commands and maintaining context awareness, it creates an intuitive shopping experience that enhances your product discovery journey.

For technical support or feature requests, please refer to the developer documentation or contact the support team.