import json
import math
import random
from collections import Counter
import os # For checking file existence

class AkinatorEfficientWeb:
    def __init__(self, dataset_source, question_selection_strategy="entropy_sampled", db_type="json"):
        self.dataset_source = dataset_source # Path for JSON, or connection info for DB
        self.db_type = db_type
        
        self.person_attributes_map = {} # Stores {name: {attributes}}
        self.all_person_names = []
        self.all_available_attributes = []
        
        self._load_data_into_memory() # Load data upon initialization

        if not self.person_attributes_map:
            raise ValueError("Dataset could not be loaded or is empty.")

        self.question_selection_strategy = question_selection_strategy
        self.attribute_sample_ratio = 0.3 
        self.min_attributes_to_sample = 5  

        # Game state variables - these will be managed per session by FastAPI
        self.probabilities = {}
        self.asked_attributes = set()
        self.questions_asked_count = 0
        
        # Game parameters (could also be configurable)
        self.certainty_threshold = 0.85
        self.min_questions_before_confident_guess = 5
        self.soft_elimination_questions_count = 3
        self.match_multiplier_soft = 1.0 
        self.mismatch_multiplier_soft = 0.4 
        self.top_k_candidates_focus = 5
        
        # Call reset_game_state to initialize for a new game
        # In a web context, this would be called when a new session starts
        # self.reset_game_state() # For standalone testing, you might call this.
                                # For FastAPI, FastAPI will manage calling this.

    def _load_data_into_memory(self):
        """Loads data from JSON or a conceptual DB into memory."""
        data_list = []
        if self.db_type == "json":
            if not os.path.exists(self.dataset_source):
                print(f"Error: JSON file not found at {self.dataset_source}")
                return
            try:
                with open(self.dataset_source, 'r') as f:
                    data_list = json.load(f) # Assuming indian_personalities_dataset_30.json
            except Exception as e:
                print(f"Error loading JSON dataset: {e}")
                return
        elif self.db_type == "db":
            # ---- DATABASE INTEGRATION POINT ----
            # Here, you would connect to your database and fetch data.
            # Example (conceptual - replace with your actual DB library e.g., psycopg2, sqlite3, SQLAlchemy):
            #
            # import sqlite3
            # conn = sqlite3.connect(self.dataset_source) # dataset_source might be 'mydb.db'
            # cursor = conn.cursor()
            # cursor.execute("SELECT name, attributes_json FROM personalities")
            # for row in cursor.fetchall():
            #     try:
            #         attributes = json.loads(row[1]) # Assuming attributes are stored as a JSON string
            #         data_list.append({"name": row[0], "attributes": attributes})
            #     except json.JSONDecodeError:
            #         print(f"Warning: Could not parse attributes for {row[0]}")
            # conn.close()
            #
            # For now, this part is conceptual. Ensure your DB query returns a list of dicts
            # similar to the JSON structure: [{"name": "Name1", "attributes": {"attr1": 1, ...}}, ...]
            print("DB loading is conceptual. For now, ensure your JSON file is used or implement DB logic.")
            # To allow testing without DB, let's fall back to JSON if db_type='db' and no data_list is formed
            if not data_list:
                 print(f"Attempting to load {self.dataset_source} as JSON as a fallback for DB.")
                 try:
                    with open(self.dataset_source, 'r') as f: data_list = json.load(f)
                 except:
                    pass # Keep data_list empty if fallback fails

        if not data_list:
            return

        self.person_attributes_map = {p['name']: p['attributes'] for p in data_list}
        self.all_person_names = list(self.person_attributes_map.keys())
        
        temp_attributes = set()
        for person_name in self.all_person_names:
            temp_attributes.update(self.person_attributes_map[person_name].keys())
        self.all_available_attributes = list(temp_attributes)

    def reset_game_state(self):
        """Resets the state for a new game session."""
        if not self.all_person_names: # Ensure data was loaded
            # This might happen if _load_data_into_memory failed silently or DB path was bad
            # For robustness in web app, might need better error handling or default dataset
            print("Warning: Attempting to reset game state but no person data is loaded.")
            self.probabilities = {}
        else:
            self.probabilities = {
                name: 1.0 / len(self.all_person_names)
                for name in self.all_person_names
            }
        self.asked_attributes = set()
        self.questions_asked_count = 0
        # print("DEBUG: Game state reset.") # For debugging

    # --- Probability and Entropy Calculation ---
    def _calculate_entropy(self, probability_values):
        if not probability_values: return 0
        probs = [p for p in probability_values if p > 1e-9]
        if not probs: return 0
        current_sum = sum(probs)
        if abs(current_sum - 1.0) > 1e-9 and current_sum > 1e-9:
            probs = [p / current_sum for p in probs]
        return -sum(p * math.log2(p) for p in probs if p > 1e-9)

    def _update_probabilities(self, attribute, user_answer):
        current_question_is_soft = self.questions_asked_count <= self.soft_elimination_questions_count

        if current_question_is_soft:
            for name in self.all_person_names:
                if self.probabilities.get(name, 0) < 1e-9: continue
                attr_val = self.person_attributes_map.get(name, {}).get(attribute, 0)
                if attr_val == user_answer:
                    self.probabilities[name] *= self.match_multiplier_soft
                else:
                    self.probabilities[name] *= self.mismatch_multiplier_soft
        else:
            for name in self.all_person_names:
                if self.probabilities.get(name, 0) < 1e-9: continue
                attr_val = self.person_attributes_map.get(name, {}).get(attribute, 0)
                if attr_val != user_answer:
                    self.probabilities[name] = 0

        current_sum_probs = sum(self.probabilities.values())
        if current_sum_probs < 1e-9 :
            return False 
        for name in self.all_person_names:
            self.probabilities[name] = self.probabilities.get(name,0) / current_sum_probs # Handle if name somehow not in probs
        return True

    # --- Question Selection ---
    def _get_top_k_candidates(self, k):
        active_probs = {name: prob for name, prob in self.probabilities.items() if prob > 1e-9}
        if not active_probs: return []
        return [item[0] for item in sorted(active_probs.items(), key=lambda item: item[1], reverse=True)[:k]]

    def _calculate_info_gain_for_subset(self, candidate_names_subset, attributes_to_evaluate):
        if not candidate_names_subset or len(candidate_names_subset) < 1: return None
        if len(candidate_names_subset) == 1 and self.questions_asked_count >= self.min_questions_before_confident_guess : return None # No IG if only one candidate and min questions met

        subset_probs_dict = {name: self.probabilities[name] for name in candidate_names_subset if name in self.probabilities and self.probabilities[name] > 1e-9}
        if not subset_probs_dict: return None
        
        subset_sum_probs = sum(subset_probs_dict.values())
        if subset_sum_probs < 1e-9: return None
            
        normalized_subset_probs_list = [p / subset_sum_probs for p in subset_probs_dict.values()]
        current_subset_entropy = self._calculate_entropy(normalized_subset_probs_list)

        best_attribute = None
        max_info_gain = -1.0 

        for attribute in attributes_to_evaluate:
            prob_ans_yes_for_subset = 0
            prob_ans_no_for_subset = 0
            subset_members_if_yes_probs = []
            subset_members_if_no_probs = []

            for name in candidate_names_subset:
                person_attr_val = self.person_attributes_map[name].get(attribute, 0)
                original_prob_for_name = subset_probs_dict[name] 

                if person_attr_val == 1:
                    prob_ans_yes_for_subset += original_prob_for_name
                    subset_members_if_yes_probs.append(original_prob_for_name)
                else:
                    prob_ans_no_for_subset += original_prob_for_name
                    subset_members_if_no_probs.append(original_prob_for_name)
            
            entropy_if_yes = self._calculate_entropy(subset_members_if_yes_probs)
            entropy_if_no = self._calculate_entropy(subset_members_if_no_probs)
            
            if subset_sum_probs < 1e-9: continue

            weight_yes = prob_ans_yes_for_subset / subset_sum_probs
            weight_no = prob_ans_no_for_subset / subset_sum_probs
            
            expected_entropy = weight_yes * entropy_if_yes + weight_no * entropy_if_no
            information_gain = current_subset_entropy - expected_entropy

            if information_gain > max_info_gain:
                max_info_gain = information_gain
                best_attribute = attribute
        
        if max_info_gain > 1e-9: 
            return best_attribute
        return None

    def _calculate_simple_heuristic_split(self, candidate_names_subset, attributes_to_evaluate):
        if not candidate_names_subset or len(candidate_names_subset) < 2: return None
        best_attribute = None
        best_min_split_score = -1 

        for attribute in attributes_to_evaluate:
            prob_mass_yes = 0
            prob_mass_no = 0
            for name in candidate_names_subset:
                person_attr_val = self.person_attributes_map[name].get(attribute, 0)
                person_prob = self.probabilities.get(name,0) # Get current prob
                if person_attr_val == 1:
                    prob_mass_yes += person_prob
                else:
                    prob_mass_no += person_prob
            
            current_score = min(prob_mass_yes, prob_mass_no) 
            if current_score > best_min_split_score:
                best_min_split_score = current_score
                best_attribute = attribute
        
        if best_attribute and best_min_split_score > 1e-9 :
            values_for_best_attr = set()
            for name_val_check in candidate_names_subset:
                values_for_best_attr.add(self.person_attributes_map[name_val_check].get(best_attribute, 0))
            if len(values_for_best_attr) > 1: return best_attribute
        return None

    def select_next_question(self):
        """Public method for FastAPI to call to get the next question."""
        # Use the same logic as _select_next_question in AkinatorProbabilisticRefined
        unasked_attributes = [attr for attr in self.all_available_attributes if attr not in self.asked_attributes]
        if not unasked_attributes:
            return None

        active_candidate_names = [name for name, prob in self.probabilities.items() if prob > 1e-9]
        if not active_candidate_names:
            return None  # No one left

        # Decide strategy based on question count
        if self.questions_asked_count > self.soft_elimination_questions_count:
            top_k_names = self._get_top_k_candidates(self.top_k_candidates_focus)
            if len(top_k_names) >= 2:  # Only focus if there's a group to differentiate
                focused_question = self._calculate_info_gain_for_subset(top_k_names, unasked_attributes)
                if focused_question:
                    return focused_question
        # Fallback or initial strategy: General information gain across all active candidates
        question = self._calculate_info_gain_for_subset(active_candidate_names, unasked_attributes)
        if question:
            return question
        # Fallback if even general IG finds nothing (e.g., all remaining are identical)
        # Pick any unasked attribute that can still vary among active candidates.
        for attr_check in unasked_attributes:
            values_for_attr = set()
            for p_name in active_candidate_names:
                values_for_attr.add(self.person_attributes_map[p_name].get(attr_check, 0))
            if len(values_for_attr) > 1:  # This attribute can still differentiate
                return attr_check
        return unasked_attributes[0] if unasked_attributes else None  # Last resort

    # --- Game State & Guessing ---
    def record_answer_and_update(self, attribute, user_answer_binary):
        """Public method for FastAPI to record an answer and update state."""
        if attribute not in self.all_available_attributes:
            return {"error": "Invalid attribute."}
        if attribute in self.asked_attributes:
             # This might happen with browser reloads, ideally client prevents re-asking same question in one flow
             # For now, we just re-update. A more robust session would just return current state.
             pass


        self.questions_asked_count += 1 # Increment for current question being processed
        self.asked_attributes.add(attribute)
        
        if not self._update_probabilities(attribute, user_answer_binary):
            return {"status": "contradiction", "message": "Your answers seem to contradict, or the person isn't in my database."}
        return {"status": "updated"}

    def _get_current_guess_info(self):
        """Returns the current best guess and its probability."""
        if not self.probabilities:
            return None, 0
        active_probs = {name: prob for name, prob in self.probabilities.items() if prob > 1e-9}
        if not active_probs:
            return None, 0
        best_candidate_name = max(active_probs, key=active_probs.get)
        max_prob = active_probs[best_candidate_name]
        return best_candidate_name, max_prob

    def get_guess_if_ready(self):
        """Public method for FastAPI to check if a guess can be made."""
        current_best_guess_name, current_max_certainty = self._get_current_guess_info()
        remaining_candidates_count = sum(1 for p in self.probabilities.values() if p > 1e-9)

        if remaining_candidates_count == 0 and self.questions_asked_count > 0:
            return {"type": "guess", "name": None, "certainty": 0, "message": "No one matches your answers."}

        if self.questions_asked_count >= self.min_questions_before_confident_guess:
            if current_max_certainty >= self.certainty_threshold or \
               (remaining_candidates_count == 1 and current_max_certainty > 0.01):
                certainty_to_display = current_max_certainty if remaining_candidates_count > 1 else 1.0
                return {"type": "guess", "name": current_best_guess_name, "certainty": certainty_to_display}
        
        if len(self.asked_attributes) >= len(self.all_available_attributes): # All questions asked
            if current_best_guess_name and current_max_certainty > 0.01:
                return {"type": "guess", "name": current_best_guess_name, "certainty": current_max_certainty, "message": "All questions asked."}
            else:
                return {"type": "guess", "name": None, "certainty": 0, "message": "Stumped after all questions."}

        return {"type": "ask_more"} # No guess yet

    def handle_wrong_guess(self, guessed_name):
        """If the user indicates the guess was wrong."""
        if guessed_name and guessed_name in self.probabilities:
            self.probabilities[guessed_name] *= 0.01 # Penalize heavily
            current_sum_probs = sum(self.probabilities.values())
            if current_sum_probs > 1e-9:
                for name_key in self.probabilities:
                    self.probabilities[name_key] /= current_sum_probs
            return True
        return False

# Example of how to run it standalone (for testing logic)
if __name__ == "__main__":
    # --- Standalone Play Function (similar to your original play(), for testing logic directly) ---
    def standalone_play(game_instance):
        print(f"Welcome to Akinator (Strategy: {game_instance.question_selection_strategy})!")
        print(f"I will ask at least {game_instance.min_questions_before_confident_guess} questions.")
        print(f"For the first {game_instance.soft_elimination_questions_count} questions, I'll use soft elimination.")
        game_instance.reset_game_state() # Initialize for this play-through

        while True:
            guess_info = game_instance.get_guess_if_ready()

            if guess_info["type"] == "guess":
                if guess_info["name"]:
                    print(f"\nI am {guess_info['certainty']*100:.1f}% sure. Are you thinking of {guess_info['name']}?")
                    final_ans = input("(yes/no): ").strip().lower()
                    if final_ans in ['yes', 'y']:
                        print("Great! I knew it!")
                    else:
                        print(f"Oh, I was mistaken about {guess_info['name']}.")
                        game_instance.handle_wrong_guess(guess_info['name'])
                        # Check if we can continue or if knowledge exhausted
                        if sum(p for p in game_instance.probabilities.values() if p > 1e-9) < 1e-8 : # Effectively all zero
                             print("It seems my knowledge is exhausted after your 'no'.")
                             break
                        if len(game_instance.asked_attributes) >= len(game_instance.all_available_attributes):
                            print("I've asked all questions and was still wrong. Well played!")
                            break
                        # else continue asking from here if game didn't end
                        if guess_info.get("message") == "All questions asked.": # Avoid infinite loop if all questions were already asked
                            break
                else:
                    print(f"\n{guess_info['message']}")
                break 
            
            # If not guessing, ask the next question
            next_attribute_to_ask = game_instance.select_next_question()

            if next_attribute_to_ask is None:
                print("\nI seem to have run out of effective questions. I'm stumped!")
                # Try a final guess based on current state if any plausible candidate
                final_name, final_cert = game_instance._get_current_guess_info()
                if final_name and final_cert > 0.01:
                     print(f"My very last attempt: is it {final_name} ({final_cert*100:.1f}%)?")
                     ans = input("(yes/no): ").strip().lower()
                     if ans in ['yes', 'y']: print("Phew!")
                     else: print("Alas!")
                break

            q_text = f"Q{game_instance.questions_asked_count + 1}: Is the person {next_attribute_to_ask.replace('_', ' ')}?"
            custom_q = {"is_male": "Is the person male?", "is_alive": "Is the person currently alive?"}
            q_text = custom_q.get(next_attribute_to_ask, q_text)

            user_response_str = ""
            while user_response_str not in ['yes', 'y', 'no', 'n']:
                user_response_str = input(f"{q_text} (yes/no/y/n): ").strip().lower()
            
            user_answer_binary = 1 if user_response_str in ['yes', 'y'] else 0
            
            update_status = game_instance.record_answer_and_update(next_attribute_to_ask, user_answer_binary)
            
            if update_status.get("status") == "contradiction":
                print(f"\n{update_status['message']}")
                break
    
    # --- Test with different strategies ---
    dataset_file = 'indian_personalities_dataset_30.json' # Make sure this file exists

    # print("\n--- Testing with 'entropy_sampled' ---")
    # game1 = AkinatorEfficientWeb(dataset_file, question_selection_strategy="entropy_sampled")
    # standalone_play(game1)

    print("\n--- Testing with 'simple_heuristic' (Recommended for resource constraints) ---")
    game2 = AkinatorEfficientWeb(dataset_file, question_selection_strategy="simple_heuristic")
    standalone_play(game2)

    # print("\n--- Testing with 'entropy_full' (Potentially slow) ---")
    # game3 = AkinatorEfficientWeb(dataset_file, question_selection_strategy="entropy_full")
    # standalone_play(game3)