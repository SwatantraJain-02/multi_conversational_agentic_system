UPDATE orders SET order_date    = order_date    + INTERVAL '28 years',
                  required_date = required_date + INTERVAL '28 years',
                  shipped_date  = shipped_date  + INTERVAL '28 years';