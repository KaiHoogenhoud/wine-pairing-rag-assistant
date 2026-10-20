"""Prompt templates and fixed answers used by the RAG pipeline.

The pipeline makes three LLM calls per question:
1. WINE_QUERY_PROMPT   - classify the question and write a wine search query
2. RECIPE_QUERY_PROMPT - turn the retrieved wine traits into a food search query
3. RAG_PROMPT          - write the answer from the retrieved context
"""

RAG_PROMPT = """You are a friendly sommelier who helps home cooks decide what to cook
with a bottle of wine they have, for example one they were given as a gift.
Answer the question using ONLY the context below. The context has two parts:
wine knowledge (grapes, regions, styles, pairing principles) and recipes.

Guidelines:
- Briefly describe the wine's key traits (body, acidity, tannin, sweetness,
  flavors) from the wine knowledge.
- Suggest one or two dishes from the recipes that pair well with it, and
  explain why in terms of those traits.
- Only suggest dishes that appear in the recipes, so the user can cook them.
- Cite the sources you used with their numbers, e.g. [1] or [3].
- If the context does not cover the wine, say so plainly. You may then
  give a cautious suggestion based on general pairing principles found in the
  context, but never invent facts about a specific wine, producer or recipe.
- Keep the answer short and practical: at most about 200 words.

Coverage check: {coverage_note}

Wine knowledge:
{wine_context}

Recipes:
{recipe_context}

Question: {question}

Answer:"""

# Step 1: classify the question and turn it into a wine-knowledge search query.
# The code uses TOPIC to route questions outside the scope and WINE to check
# whether the knowledge base actually covers that wine.
WINE_QUERY_PROMPT = """You help search a wine knowledge base (grapes, regions, wine styles,
pairing principles). Read the question and reply in exactly three lines:
TOPIC: wine if the user has a wine and asks what to eat or cook with it;
       dish if the user has a dish and asks which wine to drink with it;
       other for anything else
WINE: the grape, wine or wine region the user names, e.g. Barolo or Riesling, or none
SEARCH: the wine plus "food pairing, body, tannin, acidity", or none

Examples:
Question: I was given a bottle of Rioja, what should I cook?
TOPIC: wine
WINE: Rioja
SEARCH: Rioja food pairing, body, tannin, acidity

Question: What goes well with a dry white wine?
TOPIC: wine
WINE: none
SEARCH: dry white wine food pairing, body, acidity

Question: What wine goes with a creamy mushroom pasta?
TOPIC: dish
WINE: none
SEARCH: none

Question: What is the largest city in Italy?
TOPIC: other
WINE: none
SEARCH: none

Question: How do I fix my bike?
TOPIC: other
WINE: none
SEARCH: none

Question: {question}"""

# Step 2: turn the question into a food search query. Without it, "I have a
# Barolo, what should I cook?" searches the recipes for "Barolo", which no
# recipe mentions, and returns random dishes.
# Many wine pages never mention food (the Chablis page doesn't), so the LLM
# first names the wine's traits and then the foods that suit those traits. The
# examples map traits to foods, not wines to foods: a small model tends to copy
# an example, and copying the matching trait row is still a sensible search.
RECIPE_QUERY_PROMPT = """You help search a recipe collection to find food that pairs with wine.
Read the wine knowledge about the user's wine and reply in exactly two lines:
TRAITS: the wine's body, acidity, tannin, sweetness and main flavors
SEARCH: at most 12 words listing dishes or ingredients that suit those traits

Do not put wine or grape names in SEARCH: the recipes do not mention them.
If the wine knowledge names foods this wine pairs with, use those first.

Examples of traits and matching foods:
- light, crisp, high acidity, unoaked white -> grilled fish, shellfish, goat cheese, salad
- rich, oaked, full-bodied white -> roast chicken, creamy sauce, lobster, mushrooms
- light, fruity, low tannin red -> roast chicken, salmon, mushrooms, charcuterie
- full-bodied, high tannin red -> braised beef, roast lamb, hard cheese, stew
- off-dry, aromatic white -> spicy curry, Thai food, pork, Asian noodles
- sweet dessert wine -> fruit tart, creme brulee, blue cheese, chocolate

Wine knowledge:
{wine_context}

Question: {question}"""

# Instructions passed to RAG_PROMPT as {coverage_note}, chosen by the code.
COVERAGE_NOT_FOUND = (
    "The wine knowledge contains NO information about {wine}. Start your answer by "
    "saying that {wine} is not in your knowledge base. Do not describe {wine} or its "
    "traits. You may then suggest dishes based only on the general pairing "
    "principles in the context, and say that this is general advice."
)
COVERAGE_LOOSE = (
    "The wine knowledge only matches this question loosely. Say that your knowledge "
    "base has little on this topic and keep any suggestion cautious and general."
)
COVERAGE_OK = "The wine knowledge covers this question. Answer normally."

# Fixed answers for questions outside the assistant's scope (no LLM answer call).
EXAMPLES = (
    "Try for example: \"I was given a bottle of Chianti, what should I cook?\" "
    "or \"What dish goes with a Riesling?\""
)
OUT_OF_SCOPE_ANSWERS = {
    "dish": (
        "I work the other way around: tell me which bottle of wine you have, and I'll "
        "suggest a dish to cook with it. " + EXAMPLES
    ),
    "other": (
        "I'm a wine pairing assistant for home cooks: tell me which bottle of wine you "
        "have, and I'll suggest a dish to cook with it. " + EXAMPLES
    ),
}
