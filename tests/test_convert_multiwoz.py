import unittest

from scripts.convert_multiwoz import (
    BeliefState,
    _merge_state,
    convert_dialogue,
    extract_turn_domains,
)


def frames(
    services: list[str],
    states: list[dict],
    slots: list[dict] | None = None,
) -> dict:
    return {
        "service": services,
        "state": states,
        "slots": slots
        if slots is not None
        else [
            {
                "slot": [],
                "value": [],
                "start": [],
                "exclusive_end": [],
                "copy_from": [],
                "copy_from_value": [],
            }
            for _ in services
        ],
    }


class MultiwozConverterTests(unittest.TestCase):
    def test_first_turn_uses_state_delta_not_empty_phantom_frame(self) -> None:
        raw = frames(
            ["restaurant", "hotel"],
            [
                {
                    "active_intent": "find_restaurant",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [
                            "restaurant-area",
                            "restaurant-pricerange",
                        ],
                        "slots_values_list": [["centre"], ["expensive"]],
                    },
                },
                {
                    "active_intent": "find_hotel",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [],
                        "slots_values_list": [],
                    },
                },
            ],
        )

        self.assertEqual(extract_turn_domains(raw), ["restaurant"])

    def test_current_span_beats_accumulated_phantom_domains(self) -> None:
        previous: BeliefState = {
            "hotel-area": ("north",),
            "hotel-bookday": ("wednesday",),
            "hotel-bookpeople": ("6",),
            "hotel-bookstay": ("2",),
            "hotel-name": ("acorn guest house",),
            "hotel-parking": ("yes",),
            "hotel-pricerange": ("moderate",),
            "hotel-stars": ("4",),
        }
        raw = frames(
            ["restaurant", "hotel", "taxi"],
            [
                {
                    "active_intent": "book_restaurant",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [
                            "restaurant-bookday",
                            "restaurant-bookpeople",
                            "restaurant-booktime",
                            "restaurant-name",
                        ],
                        "slots_values_list": [
                            ["wednesday"],
                            ["6"],
                            ["19:45"],
                            ["kohinoor"],
                        ],
                    },
                },
                {
                    "active_intent": "book_hotel",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [
                            "hotel-area",
                            "hotel-bookday",
                            "hotel-bookpeople",
                            "hotel-bookstay",
                            "hotel-name",
                            "hotel-parking",
                            "hotel-pricerange",
                            "hotel-stars",
                        ],
                        "slots_values_list": [
                            ["north"],
                            ["wednesday"],
                            ["6"],
                            ["2"],
                            ["acorn guest house"],
                            ["yes"],
                            ["moderate"],
                            ["4"],
                        ],
                    },
                },
                {
                    "active_intent": "find_taxi",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [],
                        "slots_values_list": [],
                    },
                },
            ],
            [
                {
                    "slot": ["restaurant-booktime"],
                    "value": ["19:45"],
                    "start": [42],
                    "exclusive_end": [47],
                    "copy_from": [""],
                    "copy_from_value": [[]],
                },
                {
                    "slot": [],
                    "value": [],
                    "start": [],
                    "exclusive_end": [],
                    "copy_from": [],
                    "copy_from_value": [],
                },
                {
                    "slot": [],
                    "value": [],
                    "start": [],
                    "exclusive_end": [],
                    "copy_from": [],
                    "copy_from_value": [],
                },
            ],
        )

        self.assertEqual(extract_turn_domains(raw, previous), ["restaurant"])

    def test_taxi_copy_from_endpoint_slots_remain_taxi_only(self) -> None:
        raw = frames(
            ["taxi"],
            [
                {
                    "active_intent": "find_taxi",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [
                            "taxi-arriveby",
                            "taxi-departure",
                            "taxi-destination",
                        ],
                        "slots_values_list": [
                            ["19:00"],
                            ["aylesbray lodge guest house"],
                            ["don pasquale pizzeria"],
                        ],
                    },
                }
            ],
            [
                {
                    "slot": [
                        "taxi-destination",
                        "taxi-departure",
                        "taxi-arriveby",
                    ],
                    "value": ["", "", ""],
                    "start": [-1, -1, -1],
                    "exclusive_end": [-1, -1, -1],
                    "copy_from": [
                        "restaurant-name",
                        "hotel-name",
                        "restaurant-booktime",
                    ],
                    "copy_from_value": [
                        ["don pasquale pizzeria"],
                        ["aylesbray lodge guest house"],
                        ["19:00"],
                    ],
                }
            ],
        )

        self.assertEqual(extract_turn_domains(raw), ["taxi"])

    def test_dialogue_acts_override_carried_frame_state(self) -> None:
        raw = frames(
            ["hotel", "train"],
            [
                {
                    "active_intent": "book_hotel",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": ["hotel-name", "hotel-bookday"],
                        "slots_values_list": [["ashley hotel"], ["tuesday"]],
                    },
                },
                {
                    "active_intent": "find_train",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [
                            "train-departure",
                            "train-destination",
                        ],
                        "slots_values_list": [["cambridge"], ["norwich"]],
                    },
                },
            ],
        )
        acts = {
            "dialog_act": {
                "act_type": ["Train-Inform"],
                "act_slots": [
                    {
                        "slot_name": ["departure", "destination"],
                        "slot_value": ["cambridge", "norwich"],
                    }
                ],
            },
            "span_info": {
                "act_type": ["Train-Inform", "Train-Inform"],
                "act_slot_name": ["departure", "destination"],
                "act_slot_value": ["cambridge", "norwich"],
                "span_start": [50, 63],
                "span_end": [59, 70],
            },
        }

        self.assertEqual(
            extract_turn_domains(
                raw,
                {"hotel-bookday": ("tuesday",)},
                acts,
            ),
            ["train"],
        )

    def test_taxi_copy_from_endpoint_act_domains_are_removed(self) -> None:
        raw = frames(
            ["taxi", "restaurant", "hotel"],
            [
                {
                    "active_intent": "find_taxi",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": [
                            "taxi-departure",
                            "taxi-destination",
                        ],
                        "slots_values_list": [["cityroomz"], ["la mimosa"]],
                    },
                },
                {
                    "active_intent": "book_restaurant",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": ["restaurant-name"],
                        "slots_values_list": [["la mimosa"]],
                    },
                },
                {
                    "active_intent": "find_hotel",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": ["hotel-name"],
                        "slots_values_list": [["cityroomz"]],
                    },
                },
            ],
            [
                {
                    "slot": ["taxi-destination", "taxi-departure"],
                    "value": ["", ""],
                    "start": [-1, -1],
                    "exclusive_end": [-1, -1],
                    "copy_from": ["restaurant-name", "hotel-name"],
                    "copy_from_value": [["la mimosa"], ["cityroomz"]],
                },
                {
                    "slot": [],
                    "value": [],
                    "start": [],
                    "exclusive_end": [],
                    "copy_from": [],
                    "copy_from_value": [],
                },
                {
                    "slot": [],
                    "value": [],
                    "start": [],
                    "exclusive_end": [],
                    "copy_from": [],
                    "copy_from_value": [],
                },
            ],
        )
        acts = {
            "dialog_act": {
                "act_type": [
                    "Hotel-Inform",
                    "Restaurant-Inform",
                    "Taxi-Inform",
                ],
                "act_slots": [
                    {"slot_name": ["none"], "slot_value": ["none"]},
                    {"slot_name": ["none"], "slot_value": ["none"]},
                    {"slot_name": ["none"], "slot_value": ["none"]},
                ],
            },
            "span_info": {
                "act_type": [],
                "act_slot_name": [],
                "act_slot_value": [],
                "span_start": [],
                "span_end": [],
            },
        }

        self.assertEqual(extract_turn_domains(raw, dialogue_acts=acts), ["taxi"])

    def test_taxi_does_not_remove_domains_with_current_act_details(self) -> None:
        raw = frames(
            ["taxi", "hotel"],
            [
                {
                    "active_intent": "find_taxi",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": ["taxi-destination"],
                        "slots_values_list": [["cityroomz"]],
                    },
                },
                {
                    "active_intent": "find_hotel",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": ["hotel-pricerange"],
                        "slots_values_list": [["cheap"]],
                    },
                },
            ],
            [
                {
                    "slot": ["taxi-destination"],
                    "value": [""],
                    "start": [-1],
                    "exclusive_end": [-1],
                    "copy_from": ["hotel-name"],
                    "copy_from_value": [["cityroomz"]],
                },
                {
                    "slot": ["hotel-pricerange"],
                    "value": ["cheap"],
                    "start": [12],
                    "exclusive_end": [17],
                    "copy_from": [""],
                    "copy_from_value": [[]],
                },
            ],
        )
        acts = {
            "dialog_act": {
                "act_type": ["Hotel-Inform", "Taxi-Inform"],
                "act_slots": [
                    {"slot_name": ["pricerange"], "slot_value": ["cheap"]},
                    {"slot_name": ["none"], "slot_value": ["none"]},
                ],
            },
            "span_info": {
                "act_type": ["Hotel-Inform"],
                "act_slot_name": ["pricerange"],
                "act_slot_value": ["cheap"],
                "span_start": [12],
                "span_end": [17],
            },
        }

        self.assertEqual(
            extract_turn_domains(raw, dialogue_acts=acts),
            ["hotel", "taxi"],
        )

    def test_merge_state_tracks_previous_belief_values(self) -> None:
        raw = frames(
            ["hotel"],
            [
                {
                    "active_intent": "find_hotel",
                    "requested_slots": [],
                    "slots_values": {
                        "slots_values_name": ["hotel-name"],
                        "slots_values_list": [["acorn guest house"]],
                    },
                }
            ],
        )

        self.assertEqual(_merge_state(raw), {"hotel-name": ("acorn guest house",)})

    def test_thanks_plus_new_request_is_not_treated_as_closing(self) -> None:
        dialogue = {
            "turns": {
                "utterance": [
                    "The postcode is pe296fl",
                    "Thank you. I also need to find a train for Tuesday.",
                ],
                "speaker": [1, 0],
                "frames": [
                    frames([], []),
                    frames(
                        ["train"],
                        [
                            {
                                "active_intent": "find_train",
                                "requested_slots": [],
                                "slots_values": {
                                    "slots_values_name": ["train-day"],
                                    "slots_values_list": [["tuesday"]],
                                },
                            }
                        ],
                    ),
                ],
            }
        }

        self.assertEqual(convert_dialogue(dialogue)[0]["domains"], ["train"])

    def test_compound_closing_is_none(self) -> None:
        dialogue = {
            "turns": {
                "utterance": ["No, that is everything I need, thank you goodbye."],
                "speaker": [0],
                "frames": [frames([], [])],
            }
        }

        self.assertEqual(convert_dialogue(dialogue)[0]["domains"], ["none"])


if __name__ == "__main__":
    unittest.main()
