import os
import random
import uuid
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from faker import Faker

# --- Initialization ---
fake = Faker()
random.seed(42)
np.random.seed(42)

# --- User Interaction Functions ---

def ask_int(name: str, default: int) -> int:
    """
    Safely prompts the user for an integer.
    An empty or non-numeric string will result in the default value.

    Args:
        name (str): The name of the value to ask for (e.g., "users").
        default (int): The default value to return on invalid input.

    Returns:
        int: The user-provided integer or the default value.
    """
    try:
        val = input(f"How many {name} to generate? [default = {default}]: ")
        return int(val) if val.strip() else default
    except ValueError:
        print("-> Invalid input. Using the default value.")
        return default

def ask_yes_no(prompt: str, default: bool) -> bool:
    """
    Safely prompts the user for a Y/N answer.

    Args:
        prompt (str): The question to ask the user.
        default (bool): The default value to return on empty input.

    Returns:
        bool: True for 'y', False for 'n'.
    """
    while True:
        response = input(f"{prompt} (y/n) [default = {'y' if default else 'n'}]: ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        elif response == '' and default is not None:
            return default
        else:
            print("-> Invalid input. Please enter 'y' or 'n'.")

def ask_date(prompt: str, default: Optional[str] = None) -> datetime:
    """
    Safely prompts the user for a date in YYYY-MM-DD format.

    Args:
        prompt (str): The question to ask the user.
        default (Optional[str]): The default date string.

    Returns:
        datetime: The parsed datetime object.
    """
    while True:
        val = input(f"{prompt} [default = {default if default else 'not set'}]: ")
        try:
            if not val.strip() and default:
                return datetime.strptime(default, "%Y-%m-%d")
            elif val.strip():
                return datetime.strptime(val.strip(), "%Y-%m-%d")
            else:
                raise ValueError("Date is required")
        except ValueError:
            print("-> Invalid date format. Please use YYYY-MM-DD.")

# --- Data Constants ---
CATEGORIES = ["Electronics", "Home & Garden", "Toys", "Apparel", "Sports",
              "Automotive", "Books", "Beauty", "Grocery", "Pets"]
BRANDS = ["Apple", "Samsung", "Sony", "IKEA", "LEGO", "Nike", "Adidas",
          "Bosch", "Philips", "Panasonic", "JBL", "Xiaomi"]
CITIES = ["Berlin", "Hamburg", "Munich", "Cologne", "Stuttgart",
          "Frankfurt", "Düsseldorf", "Dresden", "Leipzig", "Bremen"]
LOYALTY_TIERS = ["new", "bronze", "silver", "gold", "platinum"]
EVENT_TYPES = ["view", "add_to_cart", "wishlist", "purchase"]
DEVICES = ["mobile", "desktop", "tablet"]
REFERRERS = ["organic", "search", "ad_campaign", "email", "social"]
PROMOS = ["WELCOME", "SPRING10", "SUMMER15", "AUTUMN5", "WINTER1", "PLUS"]
PAYMENT_METHODS = ["credit_card", "paypal", "apple_pay", "google_pay", "samsung_pay"]

# --- Data Generation Functions ---

def gen_products(n: int) -> pd.DataFrame:
    """Generates a DataFrame of synthetic products."""
    rows = []
    for _ in range(n):
        item_id = f"SKU{random.randint(100_000, 999_999)}"

        # Introduce some missing data to simulate real-world scenarios
        category = random.choice(CATEGORIES) if random.random() > 0.05 else None
        brand = random.choice(BRANDS) if random.random() > 0.05 else None
        title = f"{fake.word().title()} {fake.word().title()}" if random.random() > 0.05 else None

        rows.append([
            item_id,
            category if category is not None else "uncategorized",
            brand if brand is not None else "unknown",
            title if title is not None else "unknown",
            round(random.uniform(5.0, 500.0), 2),
            random.randint(0, 1000),
            f"https://picsum.photos/seed/{item_id}/400/400"
        ])
    return pd.DataFrame(rows,
                        columns=["item_id", "category", "brand", "title",
                                 "price", "stock", "image_url"])

def gen_users(n: int) -> pd.DataFrame:
    """Generates a DataFrame of synthetic users."""
    rows = []
    for _ in range(n):
        # Introduce some missing data
        gender = random.choice(["M", "F", None])
        age = random.choice([random.randint(18, 65), None])
        city = random.choice(CITIES) if random.random() > 0.02 else None
        device_pref = random.choice(DEVICES) if random.random() > 0.02 else None
        loyalty_tier = np.random.choice(LOYALTY_TIERS, p=[0.3, 0.3, 0.2, 0.15, 0.05]) if random.random() > 0.02 else None

        rows.append([
            f"U{random.randint(10_000, 99_999)}",
            fake.date_between(start_date="-3y", end_date="today"),
            gender if gender is not None else "unknown",
            age if age is not None else -1,
            city if city is not None else "unknown",
            device_pref if device_pref is not None else "unknown",
            round(abs(np.random.normal(loc=500, scale=350)), 2),
            loyalty_tier if loyalty_tier is not None else "new"
        ])
    return pd.DataFrame(rows,
                        columns=["user_id", "signup_date", "gender", "age",
                                 "city", "device_pref", "total_gmv",
                                 "loyalty_tier"])

def sample_user(df_users: pd.DataFrame) -> str:
    """Randomly samples a user_id from the users DataFrame."""
    return df_users.sample(1)["user_id"].item()

def sample_product(df_products: pd.DataFrame) -> pd.Series:
    """Randomly samples a product (row) from the products DataFrame."""
    return df_products.sample(1).iloc[0]

def gen_user_events_for_day(n: int, users: pd.DataFrame, products: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
    """Generates a DataFrame of user interaction events for a specific day."""
    rows = []
    current_day_start = datetime.combine(current_date.date(), datetime.min.time(), tzinfo=timezone.utc)
    for _ in range(n):
        prod = sample_product(products)
        # Simulate price fluctuations at the time of the event
        price_evt = round(prod["price"] * random.choice([1, 0.9, 0.8, 1.1]), 2)
        session_id = f"S_{uuid.uuid4().hex[:12]}" if random.random() > 0.05 else None

        rows.append([
            current_day_start + timedelta(seconds=random.randint(1, 24 * 3600 - 1)),
            sample_user(users),
            session_id if session_id is not None else "N/A",
            random.choices(EVENT_TYPES, weights=[0.7, 0.15, 0.1, 0.05])[0], # Weighted choice
            prod["item_id"],
            random.choice(DEVICES),
            random.choice(REFERRERS),
            price_evt
        ])
    return pd.DataFrame(rows,
                        columns=["timestamp", "user_id", "session_id",
                                 "event_type", "item_id", "device",
                                 "referrer", "price_at_event"])

def gen_orders_for_day(n: int, users: pd.DataFrame, products: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
    """Generates a DataFrame of orders for a specific day."""
    rows = []
    current_day_start = datetime.combine(current_date.date(), datetime.min.time(), tzinfo=timezone.utc)
    for _ in range(n):
        # Simulate different basket sizes
        basket = products.sample(
            np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
        )
        rows.append([
            f"O{random.randint(100_000, 999_999)}",
            sample_user(users),
            current_day_start + timedelta(seconds=random.randint(0, 24 * 3600 - 1)),
            basket["item_id"].tolist(),
            round(basket["price"].sum() * random.choice([1, 0.95, 0.9]), 2), # Simulate discounts
            random.choice(PROMOS),
            random.choice(PAYMENT_METHODS),
        ])
    return pd.DataFrame(rows,
                        columns=["order_id", "user_id", "order_ts",
                                 "item_ids", "total_amount", "promo_code",
                                 "payment_method"])

def random_query() -> str:
    """Generates a random search query or an empty string."""
    if random.random() > 0.05:
        return " ".join(fake.words(random.randint(1, 3)))
    else:
        return ""

def gen_search_logs_for_day(n: int, users: pd.DataFrame, products: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
    """Generates a DataFrame of search logs for a specific day."""
    rows = []
    current_day_start = datetime.combine(current_date.date(), datetime.min.time(), tzinfo=timezone.utc)
    for _ in range(n):
        query = random_query()
        rows.append([
            current_day_start + timedelta(seconds=random.randint(0, 24 * 3600 - 1)),
            sample_user(users),
            query if query else "",
            random.randint(1, 10),
            sample_product(products)["item_id"],
        ])
    return pd.DataFrame(rows,
                        columns=["timestamp", "user_id", "query_text",
                                 "result_click_rank", "clicked_item_id"])

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define data directories
    static_data_dir = "./synthetic_data_stream/static"
    raw_data_dir = "./synthetic_data_stream/raw"
    os.makedirs(static_data_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    users_file = os.path.join(static_data_dir, "users.csv")
    products_file = os.path.join(static_data_dir, "products.csv")

    users_df = None
    products_df = None

    # Check for existing static data and ask the user for action
    if os.path.exists(users_file) and os.path.exists(products_file):
        regenerate_static = ask_yes_no("Static data (users.csv, products.csv) already exists. Regenerate them?", False)
        if not regenerate_static:
            print("Loading existing static data...")
            users_df = pd.read_csv(users_file)
            products_df = pd.read_csv(products_file)
            
            # Fill NaNs that might occur from loading older CSVs
            users_df['gender'] = users_df['gender'].fillna('unknown')
            users_df['age'] = users_df['age'].fillna(-1)
            users_df['city'] = users_df['city'].fillna('unknown')
            users_df['device_pref'] = users_df['device_pref'].fillna('unknown')
            users_df['loyalty_tier'] = users_df['loyalty_tier'].fillna('new')
            
            products_df['category'] = products_df['category'].fillna('uncategorized')
            products_df['brand'] = products_df['brand'].fillna('unknown')
            products_df['title'] = products_df['title'].fillna('unknown')
        else:
            print("Generating new static data...")
    else:
        print("Static data not found. Generating...")

    # Generate new static data if it wasn't loaded
    if users_df is None or products_df is None:
        N_USERS_STATIC = ask_int("unique users for users.csv", 1_000)
        N_PRODUCTS_STATIC = ask_int("unique products for products.csv", 500)
        
        users_df = gen_users(N_USERS_STATIC)
        products_df = gen_products(N_PRODUCTS_STATIC)

        users_df.to_csv(users_file, index=False)
        products_df.to_csv(products_file, index=False)
        print(f"✅ Static data saved to {static_data_dir}:")
        print(f"   users.csv:    {len(users_df):>7}")
        print(f"   products.csv: {len(products_df):>7}")

    # Ask user for stream generation parameters
    NUM_EVENTS_PER_BATCH = ask_int("events (user_events) per batch", 1000)
    NUM_ORDERS_PER_BATCH = ask_int("orders per batch", 100)
    NUM_SEARCH_LOGS_PER_BATCH = ask_int("search logs per batch", 300)
    INTERVAL_SECONDS = ask_int("interval between data batches (in seconds)", 5)

    # Ask for the generation period
    print("\nSpecify the data generation period:")
    start_date = ask_date("Start date (YYYY-MM-DD)", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    end_date = ask_date("End date (YYYY-MM-DD)", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # Validate that the end date is not before the start date
    while end_date < start_date:
        print("❌ End date cannot be earlier than start date. Please try again.")
        end_date = ask_date("End date (YYYY-MM-DD)", end_date.strftime("%Y-%m-%d"))

    print(f"\nGenerating data from {start_date.date()} to {end_date.date()}...")
    print("Starting streaming data generation...")
    print("Press Ctrl+C to stop.")

    current_date = start_date
    batch_counter = 0

    # Main generation loop
    try:
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"\n--- Generating batch {batch_counter} for date: {date_str} ---")

            # Create daily directories for partitioned data
            daily_events_dir = os.path.join(raw_data_dir, "user_events", date_str)
            daily_orders_dir = os.path.join(raw_data_dir, "orders", date_str)
            daily_search_dir = os.path.join(raw_data_dir, "search_logs", date_str)
            
            os.makedirs(daily_events_dir, exist_ok=True)
            os.makedirs(daily_orders_dir, exist_ok=True)
            os.makedirs(daily_search_dir, exist_ok=True)

            # Generate batches of data
            events_batch = gen_user_events_for_day(NUM_EVENTS_PER_BATCH, users_df, products_df, current_date)
            orders_batch = gen_orders_for_day(NUM_ORDERS_PER_BATCH, users_df, products_df, current_date)
            search_batch = gen_search_logs_for_day(NUM_SEARCH_LOGS_PER_BATCH, users_df, products_df, current_date)

            # Save batches to CSV files
            events_batch.to_csv(os.path.join(daily_events_dir, f"events_batch_{batch_counter}.csv"), index=False)
            orders_batch.to_csv(os.path.join(daily_orders_dir, f"orders_batch_{batch_counter}.csv"), index=False)
            search_batch.to_csv(os.path.join(daily_search_dir, f"searches_batch_{batch_counter}.csv"), index=False)

            print(f"   user_events: {len(events_batch):>7} (saved to {daily_events_dir}/events_batch_{batch_counter}.csv)")
            print(f"   orders:      {len(orders_batch):>7} (saved to {daily_orders_dir}/orders_batch_{batch_counter}.csv)")
            print(f"   search_logs: {len(search_batch):>7} (saved to {daily_search_dir}/searches_batch_{batch_counter}.csv)")

            time.sleep(INTERVAL_SECONDS)
            current_date += timedelta(days=1)
            batch_counter += 1

        print(f"\n✅ Generation complete! Data for {batch_counter} days was generated.")

    # Handle graceful exit on KeyboardInterrupt
    except KeyboardInterrupt:
        print(f"\n❌ Data generation interrupted by user. {batch_counter} batches were generated.")
