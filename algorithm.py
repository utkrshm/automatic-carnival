import math
import json
import random
from typing import Dict, List, Literal, Tuple, Optional, Any

class Akinator:
    def __init__(self, dataset_path: str, questions_path: str, dataset_type: Literal["json", "sql"] = "json"):
        self.CERTAINTY_THRESHOLD = 0.90
        self.MIN_QUESTIONS = 5
        self.MAX_QUESTIONS = 20
        self.TOP_N_CANDIDATES = 5
        self.STRONG_MISMATCH_MULTIPLIER = 0.2
        self.STRONG_MATCH_MULTIPLIER = 1.35
        self.SOFT_MATCH_MULTIPLIER = 1.1
        self.SOFT_MISMATCH_MULTIPLIER = 0.5
        
        self.dataset = self._load_data(dataset_path, dataset_type)
        if not self.dataset:
            raise ValueError("Dataset is empty or not loaded correctly.")
        
        self.people = [person["name"] for person in self.dataset]
        self.attrs = self._get_attrs()
        self.people_attrs_map = {person["name"]: person["attributes"] for person in self.dataset}
        self.questions = self._get_questions(questions_path)
        
        print(f"Building an Akinator game with {len(self.people)} people having {len(self.attrs)} traits...\n")
        self._reset()

    def _load_data(self, dataset_path: str, dataset_type: Literal["json", "sql"]):
        if dataset_type.lower() == "json":
            try:
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return []
        
        elif dataset_type == "sql":
            print("SQL loading not implemented.")
            return []
        
        else:
            raise ValueError("Unsupported dataset type. Use 'json' or 'sql'.")

    def _get_questions(self, questions_path: str) -> Dict[str, str]:
        try:
            with open(questions_path, 'r') as f:
                questions = json.load(f)
            return questions
        
        except Exception as e:
            print(f"Error loading questions: {e}")
            return {}

    def _get_attrs(self) -> List[str]:
        attrs = set()
        
        for person in self.dataset:
            attrs.update(person["attributes"].keys())
        
        return list(attrs)

    def _reset(self):
        self.probabilities = {person: 1 / len(self.people) for person in self.people}
        self.asked_attrs = set()
        self.n_questions_asked = 0
        self.RANDOMNESS = 0.5
        self.RETRY = False

    def _calc_entropy(self, probs: List[float]) -> float:
        if not probs or sum(probs) < 1e-9:
            return 0.0
        
        sum_probs = sum(probs)
        if abs(sum_probs - 1.0) > 1e-9 and len(probs) > 1:  # Normalize if sum > 1
            probs = [p / sum_probs for p in probs]
        
        return -sum(p * math.log2(p) for p in probs if p > 1e-9)

    def _get_top_candidates(self, n: int = 0) -> List[Tuple[str, float]]:
        if not self.probabilities:
            return []
        
        if not n:
            n = self.TOP_N_CANDIDATES
        
        sorted_probs = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:n]

    def _calc_info_gain_subset(self, subset_candidates: List[str], unasked_attrs: List[str]) -> Optional[str]:
        if not subset_candidates or len(subset_candidates) < 1:
            return None
        
        subset_probs = {name: self.probabilities[name] for name in subset_candidates if name in self.probabilities and self.probabilities[name] > 1e-9}
        subset_sum = sum(subset_probs.values())
        if subset_sum < 1e-9:
            return None
        
        norm_probs = [p / subset_sum for p in subset_probs.values()]
        subset_entropy = self._calc_entropy(norm_probs)
        
        best_attr = None
        max_gain = -1.0
        
        for attr in unasked_attrs:
            if attr in self.asked_attrs:    # Double check if attr is already asked
                continue
            
            people_with_yes_prob = []
            people_with_no_prob = []
            sum_yes = 0.0
            sum_no = 0.0
            
            for person in subset_candidates:
                value = self.people_attrs_map[person].get(attr, 0)
                prob = subset_probs.get(person, 0)
                
                if value == 1:
                    people_with_yes_prob.append(prob)
                    sum_yes += prob
                elif value == 0:
                    people_with_no_prob.append(prob)
                    sum_no += prob
            
            weight_yes = sum_yes / subset_sum if subset_sum > 1e-9 else 0
            weight_no = sum_no / subset_sum if subset_sum > 1e-9 else 0
            
            entropy_yes = self._calc_entropy(people_with_yes_prob)
            entropy_no = self._calc_entropy(people_with_no_prob)
            
            conditional_entropy = weight_yes * entropy_yes + weight_no * entropy_no
            info_gain = subset_entropy - conditional_entropy
            
            if info_gain > max_gain:
                max_gain = info_gain
                best_attr = attr
        
        if max_gain > 1e-9 and best_attr is not None:
            return best_attr
        
        return None

    def _calc_info_gain_focused(self, top_n_names: List[str], unasked_attrs: List[str]) -> Optional[str]:
        return self._calc_info_gain_subset(top_n_names, unasked_attrs)

    def _calc_info_gain_general(self, unasked_attrs: List[str]) -> Optional[str]:
        active_names = [name for name, prob in self.probabilities.items() if prob > 1e-9]
        return self._calc_info_gain_subset(active_names, unasked_attrs)

    def _get_current_guess(self) -> Tuple[Optional[str], float]:
        if not self.probabilities:
            return None, 0.0
        
        sorted_probs = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        if not sorted_probs:
            return None, 0.0
        
        best_guess, best_prob = sorted_probs[0]
        
        return best_guess, best_prob

    def _update_probs(self, attr: str, answer: float) -> bool:
        for person in list(self.probabilities.keys()):
            value = self.people_attrs_map[person].get(attr, 0)
            
            # Valid for when answer is [0, 1] because attribute is also going to be [0, 1]
            # (Strong match/mismatch for exact answers)
            if value == answer:
                self.probabilities[person] *= self.STRONG_MATCH_MULTIPLIER
            elif abs(value - answer) == 1:
                self.probabilities[person] *= self.STRONG_MISMATCH_MULTIPLIER
            
            # Valid for when answer is [0.25, 0.75] because attribute is going to be [0, 1] so
            # difference is answer and value is going to be either 0.25 (match) or 0.75 (mismatch)
            # (Soft match/mismatch for "probably" answers)
            elif abs(value - answer) < 0.5:
                self.probabilities[person] *= self.SOFT_MATCH_MULTIPLIER
            elif abs(value - answer) > 0.5:
                self.probabilities[person] *= self.SOFT_MISMATCH_MULTIPLIER
        
        current_sum = sum(self.probabilities.values())
        if current_sum < 1e-9:  return False
        
        # Normalize probabilities
        for name in self.probabilities:
            if self.probabilities.get(name, 0) > 1e-9:
                self.probabilities[name] /= current_sum
            else:
                self.probabilities[name] = 0.0
        
        return True

    def select_next_question(self) -> Optional[str]:
        if self.RETRY:
            self.RANDOMNESS = 0.0
        elif self.n_questions_asked > self.MIN_QUESTIONS and not self.RETRY:
            self.RANDOMNESS = 0.1
        
        unasked_attrs = [attr for attr in self.attrs if attr not in self.asked_attrs]
        if not unasked_attrs:
            return None
        
        if self.RANDOMNESS > 0 and len(unasked_attrs) > 1:
            sample_size = max(1, int((1 - self.RANDOMNESS) * len(unasked_attrs)))
            sample_size = min(sample_size, len(unasked_attrs))
            if sample_size < len(unasked_attrs):
                unasked_attrs = random.sample(unasked_attrs, sample_size)
        
        active_names = [name for name, prob in self.probabilities.items() if prob > 1e-9]
        if not active_names:
            return None
        
        next_attr = None
        if self.n_questions_asked >= self.MIN_QUESTIONS:
            top_n_names = [name for name, prob in self._get_top_candidates() if prob > 1e-9]

            if len(top_n_names) > 1:
                # Use focused information gain
                next_attr = self._calc_info_gain_focused(top_n_names, unasked_attrs)

        # If no attribute is found from focused information gain, use general information gain
        if not next_attr:
            next_attr = self._calc_info_gain_general(unasked_attrs)
        
        if not next_attr and unasked_attrs:
            next_attr = unasked_attrs[0]
        
        return next_attr

    def get_question_text(self, attribute_key: str) -> str:
        q_text = self.questions.get(attribute_key, f"Is the person {attribute_key.replace('_', ' ')}?")
        
        # Special logic for handling nickname
        if attribute_key.startswith("nickname_"):
            nickname = attribute_key.replace("nickname_", "").replace("_", " ")
            if nickname != "None":
                q_text = "Is the person's nickname " + nickname + "?"
        
        return f"Q{self.n_questions_asked + 1}: {q_text}"
        
    # --- Backend methods ---
    def start_game(self) -> Dict[str, Any]:
        self._reset()
        next_attribute = self.select_next_question()
        
        if next_attribute:
            return {
                "status": "playing",
                "question_text": self.get_question_text(next_attribute),
                "attribute_key": next_attribute,
                "questions_asked": self.n_questions_asked,
            }
        else:
            return {
                "status": "error",
                "message": "Cannot start game: No questions available or no candidates.",
                "guess": None,
                "certainty": 0.0,
            }

    def process_answer(self, attribute_key: str, answer_numeric: float) -> Dict[str, Any]:
        if attribute_key in self.asked_attrs:
            return {"status": "error", "message": "Attribute already asked."}
        
        self.asked_attrs.add(attribute_key)
        self.n_questions_asked += 1
        self.RETRY = False
        
        if not self._update_probs(attribute_key, answer_numeric):
            return {"status": "failure", "message": "You beat me! I couldn't guess.", "guess": None, "certainty": 0.0}
        
        current_guess_name, current_certainty = self._get_current_guess()
        remaining_candidates_count = sum(1 for p in self.probabilities.values() if p > 1e-9)
        
        if self.n_questions_asked >= self.MIN_QUESTIONS:
            if current_certainty >= self.CERTAINTY_THRESHOLD or (remaining_candidates_count == 1 and current_certainty > 0.1):
                return {
                    "status": "make_guess",
                    "guess": current_guess_name,
                    "certainty": current_certainty,
                }
        
        if self.n_questions_asked >= len(self.attrs) or self.n_questions_asked >= self.MAX_QUESTIONS:
            return {
                "status": "failure",
                "message": "You beat me! I couldn't guess.",
                "guess": current_guess_name,
                "certainty": current_certainty,
            }
        
        next_attribute = self.select_next_question()
        
        if next_attribute:
            return {
                "status": "playing",
                "question_text": self.get_question_text(next_attribute),
                "attribute_key": next_attribute,
                "questions_asked": self.n_questions_asked,
            }
        else:
            return {
                "status": "failure",
                "message": "You beat me! I couldn't guess.",
                "guess": current_guess_name,
                "certainty": current_certainty,
            }

    def process_mistaken_guess(self, wrong_guess_name: str) -> Dict[str, Any]:
        if wrong_guess_name in self.probabilities:
            self.probabilities[wrong_guess_name] *= 0.01
            current_sum = sum(p for p in self.probabilities.values() if p > 1e-9)
        
            if current_sum > 1e-9:
                for name in self.probabilities:
                    if self.probabilities[name] > 1e-9:
                        self.probabilities[name] /= current_sum
                    else:
                        self.probabilities[name] = 0.0
        
            else:
                return {"status": "failure", "message": "You beat me! I couldn't guess.", "guess": None, "certainty": 0.0}
        
        self.RETRY = True
        next_attribute = self.select_next_question()
        
        if next_attribute:
            return {
                "status": "playing",
                "question_text": self.get_question_text(next_attribute),
                "attribute_key": next_attribute,
                "questions_asked": self.n_questions_asked,
            }
        
        else:
            current_guess_name, current_certainty = self._get_current_guess()
            return {
                "status": "failure",
                "message": "You beat me! I couldn't guess.",
                "guess": current_guess_name,
                "certainty": current_certainty,
            }

    # --- State Management for Database Persistence ---
    def get_state(self) -> Dict[str, Any]:
        """Serializes the current dynamic game state to a JSON-compatible dictionary."""
        return {
            "probabilities": self.probabilities,
            "asked_attrs": list(self.asked_attrs), # Convert set to list for JSON
            "n_questions_asked": self.n_questions_asked,
            "RANDOMNESS": self.RANDOMNESS,
            "RETRY": self.RETRY,
        }

    def _load_state(self, state: Dict[str, Any]):
        """Restores the dynamic game state from a dictionary."""
        self.probabilities = state.get("probabilities", {})
        self.asked_attrs = set(state.get("asked_attrs", [])) # Convert list back to set
        self.n_questions_asked = state.get("n_questions_asked", 0)
        self.RANDOMNESS = state.get("RANDOMNESS", 0.5) # Default if not in state
        self.RETRY = state.get("RETRY", False) # Default if not in state
        print(f"Game state loaded. Questions asked: {self.n_questions_asked}, Retry: {self.RETRY}")

    # Function for standalone testing in CLI
    def play(self):
        print("Welcome to Akinator (Mistake Tolerant Version with Probably and Probably Not)!")
        print(f"I will ask at least {self.MIN_QUESTIONS} questions.")
        print("I will try to guess even if you make a mistake or two. Answer with 'yes' (y), 'no' (n), 'probably' (p), or 'probably not' (pn).")
        self._reset()
        while True:
            current_best_guess_name, current_max_certainty = self._get_current_guess()
            remaining_candidates_count = sum(1 for p in self.probabilities.values() if p > 1e-9)
            if remaining_candidates_count == 0 and self.n_questions_asked > 0:
                print("\nHmm, based on your answers, I don't think the person is in my database or there's a contradiction.")
                break
            made_a_guess_this_turn = False
            if self.n_questions_asked >= self.MIN_QUESTIONS:
                if current_max_certainty >= self.CERTAINTY_THRESHOLD or (remaining_candidates_count == 1 and current_max_certainty > 0.1):
                    certainty_to_display = current_max_certainty if remaining_candidates_count > 1 else self.probabilities.get(current_best_guess_name, 1.0)
                    print(f"\nI am {certainty_to_display*100:.1f}% sure. Are you thinking of {current_best_guess_name}?")
                    final_ans = input("(yes/no): ").strip().lower()
                    made_a_guess_this_turn = True
                    if final_ans in ['yes', 'y']:
                        print("Great! I knew it!")
                        return
                    else:
                        print(f"Oh, I was mistaken about {current_best_guess_name}. Let me try again.")
                        if current_best_guess_name in self.probabilities:
                            self.probabilities[current_best_guess_name] *= 0.01
                            current_sum_probs = sum(p for p in self.probabilities.values() if p > 1e-9)
                            if current_sum_probs > 1e-9:
                                for name_key in self.probabilities:
                                    if self.probabilities[name_key] > 1e-9:
                                        self.probabilities[name_key] /= current_sum_probs
                                    else:
                                        self.probabilities[name_key] = 0.0
                            else:
                                print("It seems my knowledge is exhausted or answers are inconsistent.")
                                return
            if self.n_questions_asked >= len(self.attrs) and not made_a_guess_this_turn:
                if current_best_guess_name and current_max_certainty > 0.01:
                    print(f"\nI've asked all I can. My best guess is {current_best_guess_name} (Certainty: {current_max_certainty*100:.1f}%). Is it them?")
                    final_ans = input("(yes/no): ").strip().lower()
                    if final_ans in ['yes', 'y']:
                        print("Excellent!")
                    else:
                        print("Hmm, I couldn't quite get it. Well played!")
                else:
                    print("\nI'm completely stumped after asking all questions! I couldn't find a match.")
                break
            next_attribute_to_ask = self.select_next_question()
            if next_attribute_to_ask is None:
                if not made_a_guess_this_turn:
                    if current_best_guess_name and current_max_certainty > 0.01:
                        print(f"\nI've run out of good questions. My best guess is {current_best_guess_name} (Certainty: {current_max_certainty*100:.1f}%). Is it them?")
                        final_ans = input("(yes/no): ").strip().lower()
                        if final_ans in ['yes', 'y']:
                            print("Excellent!")
                        else:
                            print("Hmm, I couldn't quite get it. Well played!")
                    else:
                        print("\nI'm stumped! I couldn't find a match or a good question to ask.")
                break
            self.n_questions_asked += 1
            q_text = self.get_question_text(next_attribute_to_ask)
            user_response_str = ""
            while user_response_str not in ['yes', 'y', 'no', 'n', 'probably', 'p', 'probably not', 'pn']:
                user_response_str = input(f"{q_text} (yes/no/probably/probably not/y/n/p/pn): ").strip().lower()
            usr_ans_dict = {'yes': 1, 'y': 1, 'no': 0, 'n': 0, 'probably': 0.75, 'p': 0.75, 'probably not': 0.25, 'pn': 0.25}
            user_answer_numeric = usr_ans_dict[user_response_str]
            self.asked_attrs.add(next_attribute_to_ask)
            if not self._update_probs(next_attribute_to_ask, user_answer_numeric):
                print("\nHmm, based on your answers, it seems there's a contradiction or the person isn't in my database.")
                break
            sorted_probs = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)
            print("\n--- Current Top 5 Candidates ---")
            for name, prob in sorted_probs[:5]:
                if prob > 0.01:
                    print(f"{name}: {prob*100:.1f}%")


if __name__ == "__main__":
    dataset_path = "data/characters_data.json"  # Path to your dataset
    questions_path = "data/questions.json"  # Path to your questions
    game = Akinator(dataset_path, questions_path)
    game.play()