from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

PIZZA_PRICES = {
    "cheese": 8.0,
    "pepperoni": 10.0,
    "margherita": 9.0,
    "hawaiian": 11.0,
    "vegetarian": 10.0,
    "vegan": 12.0,
    "gluten-free": 13.0,
    "bbq-chicken": 12.0,
    "buffalo-chicken": 12.0,
}

def parse_quantity(value: Any) -> int:
    if value is None:
        return 1

    normalized = str(value).strip().lower()
    if normalized in NUMBER_WORDS:
        return NUMBER_WORDS[normalized]

    try:
        return int(float(normalized))
    except ValueError:
        return 1


class ActionConfirmPizzaOrder(Action):
    def name(self) -> Text:
        return "action_confirm_pizza_order"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        pizza_type = tracker.get_slot("selected_pizza")
        quantity = parse_quantity(tracker.get_slot("pizza_quantity"))
        pizza_type = str(pizza_type).strip().lower().replace(" ", "-").replace("_", "-")

        if pizza_type not in PIZZA_PRICES:
            dispatcher.utter_message(
                text="Sorry, I can only order pizzas from the available menu."
            )
            return []

        unit_price = PIZZA_PRICES.get(pizza_type, 0)
        total = unit_price * quantity
        plural = "pizza" if quantity == 1 else "pizzas"

        dispatcher.utter_message(
            text=(
                f"Great choice. I can help you order {quantity} "
                f"{pizza_type} {plural}. The total is ${total:.2f}."
            )
        )

        return [
            SlotSet("selected_pizza", pizza_type),
            SlotSet("pizza_quantity", quantity),
        ]
