-- Shopping Cart Database Schema
-- Creates tables and indexes for shopping cart functionality

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Shopping cart table
CREATE TABLE IF NOT EXISTS shopping_cart (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    product_title VARCHAR(500) NOT NULL,
    product_price DECIMAL(10,2),
    product_image_url VARCHAR(1000),
    quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    product_metadata JSONB DEFAULT '{}'::jsonb,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, product_id)
);

-- Cart session summary table for quick access to cart totals
CREATE TABLE IF NOT EXISTS cart_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    total_items INTEGER DEFAULT 0 CHECK (total_items >= 0),
    total_value DECIMAL(10,2) DEFAULT 0.00 CHECK (total_value >= 0),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Intent classification cache table for router optimization
CREATE TABLE IF NOT EXISTS intent_classifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_text TEXT NOT NULL,
    classified_intent VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    context_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for shopping_cart table
CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_id ON shopping_cart(session_id);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_product_id ON shopping_cart(product_id);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_updated_at ON shopping_cart(updated_at);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_added_at ON shopping_cart(added_at);
CREATE INDEX IF NOT EXISTS idx_shopping_cart_session_product ON shopping_cart(session_id, product_id);

-- Indexes for cart_sessions table
CREATE INDEX IF NOT EXISTS idx_cart_sessions_last_updated ON cart_sessions(last_updated);
CREATE INDEX IF NOT EXISTS idx_cart_sessions_total_items ON cart_sessions(total_items);

-- Indexes for intent_classifications table
CREATE INDEX IF NOT EXISTS idx_intent_classifications_message_hash ON intent_classifications(md5(message_text));
CREATE INDEX IF NOT EXISTS idx_intent_classifications_context_hash ON intent_classifications(context_hash);
CREATE INDEX IF NOT EXISTS idx_intent_classifications_created_at ON intent_classifications(created_at);
CREATE INDEX IF NOT EXISTS idx_intent_classifications_intent ON intent_classifications(classified_intent);

-- Function to update updated_at timestamp for shopping_cart
CREATE OR REPLACE FUNCTION update_shopping_cart_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to update cart_sessions when shopping_cart changes
CREATE OR REPLACE FUNCTION update_cart_session_summary()
RETURNS TRIGGER AS $$
BEGIN
    -- Handle INSERT and UPDATE
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        INSERT INTO cart_sessions (session_id, total_items, total_value, last_updated)
        SELECT 
            NEW.session_id,
            COALESCE(SUM(quantity), 0),
            COALESCE(SUM(quantity * COALESCE(product_price, 0)), 0),
            CURRENT_TIMESTAMP
        FROM shopping_cart 
        WHERE session_id = NEW.session_id
        ON CONFLICT (session_id) 
        DO UPDATE SET
            total_items = EXCLUDED.total_items,
            total_value = EXCLUDED.total_value,
            last_updated = EXCLUDED.last_updated;
        
        RETURN NEW;
    END IF;
    
    -- Handle DELETE
    IF TG_OP = 'DELETE' THEN
        INSERT INTO cart_sessions (session_id, total_items, total_value, last_updated)
        SELECT 
            OLD.session_id,
            COALESCE(SUM(quantity), 0),
            COALESCE(SUM(quantity * COALESCE(product_price, 0)), 0),
            CURRENT_TIMESTAMP
        FROM shopping_cart 
        WHERE session_id = OLD.session_id
        ON CONFLICT (session_id) 
        DO UPDATE SET
            total_items = EXCLUDED.total_items,
            total_value = EXCLUDED.total_value,
            last_updated = EXCLUDED.last_updated;
        
        -- If no items left, keep the session record but with zero values
        INSERT INTO cart_sessions (session_id, total_items, total_value, last_updated)
        VALUES (OLD.session_id, 0, 0, CURRENT_TIMESTAMP)
        ON CONFLICT (session_id) 
        DO UPDATE SET
            total_items = 0,
            total_value = 0,
            last_updated = CURRENT_TIMESTAMP
        WHERE NOT EXISTS (SELECT 1 FROM shopping_cart WHERE session_id = OLD.session_id);
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp and summary updates
CREATE TRIGGER update_shopping_cart_updated_at_trigger
    BEFORE UPDATE ON shopping_cart 
    FOR EACH ROW 
    EXECUTE FUNCTION update_shopping_cart_updated_at();

CREATE TRIGGER update_cart_session_summary_trigger
    AFTER INSERT OR UPDATE OR DELETE ON shopping_cart
    FOR EACH ROW
    EXECUTE FUNCTION update_cart_session_summary();

-- Function to clean up old intent classifications (for maintenance)
CREATE OR REPLACE FUNCTION cleanup_old_intent_classifications(max_age_days INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM intent_classifications 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * max_age_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Function to get cart summary for a session
CREATE OR REPLACE FUNCTION get_cart_summary(p_session_id VARCHAR(255))
RETURNS TABLE(
    session_id VARCHAR(255),
    total_items INTEGER,
    total_value DECIMAL(10,2),
    item_count INTEGER,
    last_updated TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.session_id,
        cs.total_items,
        cs.total_value,
        COUNT(sc.id)::INTEGER as item_count,
        cs.last_updated
    FROM cart_sessions cs
    LEFT JOIN shopping_cart sc ON cs.session_id = sc.session_id
    WHERE cs.session_id = p_session_id
    GROUP BY cs.session_id, cs.total_items, cs.total_value, cs.last_updated;
END;
$$ language 'plpgsql';