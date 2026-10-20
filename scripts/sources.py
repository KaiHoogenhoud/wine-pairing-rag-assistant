"""Curated page lists for the knowledge base.

Each entry is (page title, category). The category is stored as metadata in the
downloaded file so retrieval can later filter on it (e.g. only pairing theory,
only recipes). Titles that redirect are resolved automatically; titles that no
longer exist are skipped with a warning.

Edit these lists to grow or shrink the corpus, then re-run
`python scripts/download_data.py`.
"""

# --- Wikipedia (CC BY-SA 4.0) -------------------------------------------------

WIKIPEDIA_PAGES = [
    # Pairing theory and how wine is described
    ("Wine and food pairing", "pairing_theory"),
    ("Wine tasting descriptors", "pairing_theory"),
    ("Acids in wine", "pairing_theory"),
    ("Sweetness of wine", "pairing_theory"),
    ("Oak (wine)", "pairing_theory"),
    # Wine styles
    ("Red wine", "wine_style"),
    ("White wine", "wine_style"),
    ("Rosé", "wine_style"),
    ("Orange wine", "wine_style"),
    ("Sparkling wine", "wine_style"),
    ("Dessert wine", "wine_style"),
    ("Fortified wine", "wine_style"),
    # Red grape varieties
    ("Cabernet Sauvignon", "grape_red"),
    ("Merlot", "grape_red"),
    ("Pinot noir", "grape_red"),
    ("Syrah", "grape_red"),
    ("Grenache", "grape_red"),
    ("Malbec", "grape_red"),
    ("Zinfandel", "grape_red"),
    ("Sangiovese", "grape_red"),
    ("Nebbiolo", "grape_red"),
    ("Tempranillo", "grape_red"),
    ("Gamay", "grape_red"),
    ("Cabernet Franc", "grape_red"),
    ("Barbera", "grape_red"),
    # White grape varieties
    ("Chardonnay", "grape_white"),
    ("Sauvignon blanc", "grape_white"),
    ("Riesling", "grape_white"),
    ("Pinot gris", "grape_white"),
    ("Chenin blanc", "grape_white"),
    ("Gewürztraminer", "grape_white"),
    ("Viognier", "grape_white"),
    ("Albariño", "grape_white"),
    ("Grüner Veltliner", "grape_white"),
    ("Sémillon", "grape_white"),
    ("Muscat (grape)", "grape_white"),
    # Well-known wines and regions
    ("Champagne", "wine_region"),
    ("Prosecco", "wine_region"),
    ("Cava (Spanish wine)", "wine_region"),
    ("Port wine", "wine_region"),
    ("Sherry", "wine_region"),
    ("Sauternes (wine)", "wine_region"),
    ("Chianti", "wine_region"),
    ("Rioja DOCa", "wine_region"),
    ("Beaujolais", "wine_region"),
    ("Burgundy wine", "wine_region"),
    ("Bordeaux wine", "wine_region"),
    ("Barolo", "wine_region"),
    ("Chablis (AOC)", "wine_region"),
]

# --- Wikibooks Cookbook (CC BY-SA 4.0) ----------------------------------------
# Home-cooking dishes with a clear wine-pairing angle: a mix of proteins,
# cuisines, spice levels and desserts so the assistant sees varied cases.

WIKIBOOKS_RECIPES = [
    # Beef, lamb and pork
    ("Cookbook:Boeuf Bourguignon", "recipe"),
    ("Cookbook:Traditional Beef Stroganoff", "recipe"),
    ("Cookbook:Beef Stew I", "recipe"),
    ("Cookbook:Classic Pot Roast", "recipe"),
    ("Cookbook:Chili con Carne", "recipe"),
    ("Cookbook:Goulash", "recipe"),
    ("Cookbook:Beef Fajitas", "recipe"),
    ("Cookbook:Shepherd's Pie I", "recipe"),
    ("Cookbook:Greek Moussaka", "recipe"),
    ("Cookbook:Roast Leg of Lamb with Gravy", "recipe"),
    ("Cookbook:Barbecue Ribs", "recipe"),
    ("Cookbook:Pulled Pork Sandwiches", "recipe"),
    # Poultry
    ("Cookbook:Roast Chicken", "recipe"),
    ("Cookbook:Coq au Vin I", "recipe"),
    ("Cookbook:Chicken Cacciatore", "recipe"),
    ("Cookbook:Duck à l'Orange", "recipe"),
    ("Cookbook:Chicken Curry", "recipe"),
    ("Cookbook:Chicken Tikka Masala", "recipe"),
    ("Cookbook:Indian Butter Chicken I", "recipe"),
    ("Cookbook:Thai Green Curry with Chicken", "recipe"),
    ("Cookbook:Pad Thai", "recipe"),
    # Fish and seafood
    ("Cookbook:Fish and Chips", "recipe"),
    ("Cookbook:Bouillabaisse", "recipe"),
    ("Cookbook:Cioppino (Italian Seafood Stew)", "recipe"),
    ("Cookbook:Mussels in Onion and Butter Sauce (Moules Mariniere)", "recipe"),
    ("Cookbook:Gambas al Ajillo (Garlic Shrimp)", "recipe"),
    ("Cookbook:Egyptian Fish Tagine (Samak Tajine)", "recipe"),
    ("Cookbook:Paella de Marisco", "recipe"),
    ("Cookbook:Salmon Spaghetti", "recipe"),
    # Pasta, rice and pizza
    ("Cookbook:Spaghetti alla Carbonara", "recipe"),
    ("Cookbook:Spaghetti with Bolognese Meat Sauce", "recipe"),
    ("Cookbook:Lasagne", "recipe"),
    ("Cookbook:Pesto", "recipe"),
    ("Cookbook:Risotto ai Funghi", "recipe"),
    ("Cookbook:Risotto alla Milanese", "recipe"),
    ("Cookbook:New York City–Style Pizza", "recipe"),
    ("Cookbook:Macaroni and Cheese", "recipe"),
    # Vegetarian, cheese and starters
    ("Cookbook:Ratatouille I", "recipe"),
    ("Cookbook:Quiche Lorraine", "recipe"),
    ("Cookbook:Fondue Neuchateloise", "recipe"),
    ("Cookbook:French Onion Soup", "recipe"),
    ("Cookbook:Gazpacho", "recipe"),
    ("Cookbook:Caesar Salad", "recipe"),
    ("Cookbook:Greek Salad", "recipe"),
    ("Cookbook:Hummus", "recipe"),
    ("Cookbook:Falafel", "recipe"),
    # Desserts
    ("Cookbook:Tiramisù", "recipe"),
    ("Cookbook:Crème Brûlée", "recipe"),
    ("Cookbook:Chocolate Mousse", "recipe"),
    ("Cookbook:Apple Pie I", "recipe"),
    ("Cookbook:New York Cheesecake", "recipe"),
]
