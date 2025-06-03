import random
import re



def format_reward(completion):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    while the final answer is enclosed within <answer> and </answer> tags."""

    pattern1 = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    match1 = re.match(pattern1, completion, re.DOTALL | re.MULTILINE)

    pattern2 = r"^<think>\n.*?\n</think>\n<answer>\s.*?\s</answer>$"
    match2 = re.match(pattern2, completion, re.DOTALL | re.MULTILINE)

    pattern3 = r"^<think>\s.*?\s</think>\n<answer>\n.*?\n</answer>$"
    match3 = re.match(pattern3, completion, re.DOTALL | re.MULTILINE)

    pattern4 = r"^<think>\s.*?\s</think>\n<answer>\s.*?\s</answer>$"
    match4 = re.match(pattern4, completion, re.DOTALL | re.MULTILINE)
    if match1 or match2 or match3 or match4:
        return 0.0
    else:
        return -1.0


def tag_count_reward(completion):
   """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`."""
   count = 0.0

   # Check for opening think tags (using regex to capture both formats)
   think_start_pattern = r"<think>\n|<think>\s"
   think_start_matches = re.findall(think_start_pattern, completion)
   if len(think_start_matches) != 1:
       count -= 0.25

   # Check for closing think tags
   think_end_pattern = r"\n</think>\n|\s</think>\n"
   think_end_matches = re.findall(think_end_pattern, completion)
   if len(think_end_matches) != 1:
       count -= 0.25

   # Check for opening answer tags
   answer_start_pattern = r"\n<answer>\n|\n<answer>\s"
   answer_start_matches = re.findall(answer_start_pattern, completion)
   if len(answer_start_matches) != 1:
       count -= 0.25

   # Check for closing answer tags
   answer_end_pattern = r"\n</answer>|\s</answer>"
   answer_end_matches = re.findall(answer_end_pattern, completion)
   if len(answer_end_matches) != 1:
       count -= 0.25

   return count

def extract_solution(solution_str):
    """
    Combined extraction function that handles both strict/flexible formats and multiple answer tags.

    Processing:
    1. Try strict format matching first (requires exact newlines)
    2. Fall back to flexible format matching (allows any whitespace)
    3. If multiple matches are found, return the last one

    Args:
        solution_str: String containing the answer(s)

    Returns:
        Extracted answer, or None if no valid format matches
    """
    # Check for multiple matches (from second function)
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    # If there are 2 or more matches, return the last one
    if len(matches) >= 2:
        return matches[-1].group(1).strip()

    # Try strict format matching (from first function)
    strict_pattern = r"<answer>\n(.*?)\n</answer>"
    strict_match = re.search(strict_pattern, solution_str, re.DOTALL)

    if strict_match:
        return strict_match.group(1)

    # If strict matching fails, try flexible format matching
    flexible_pattern = r"<answer>\s*(.*?)\s*</answer>"
    flexible_match = re.search(flexible_pattern, solution_str, re.DOTALL)

    if flexible_match:
        return flexible_match.group(1).strip()

    # If no format matches, return None
    return None

def exclude_reward(completions, ground_truth):

    response_list= completions.split('\n')[0]
    candidate_text_list = ground_truth['candidate_text'].tolist()
    if response_list not in candidate_text_list:
        reward=-0.5
    else:
        final_response =response_list
        if final_response == ground_truth['gt']:
            reward = 0
        else:
            reward = 1
    return reward


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    equation = extract_solution(solution_str=solution_str)

    format_r=format_reward(solution_str)
    tag_r=tag_count_reward(solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        print(f"No equation found")
        return -1+format_r+tag_r
    else:
        return exclude_reward(equation, ground_truth)+format_r+tag_r
