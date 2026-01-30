"""
Text-Based Reasoning Shift Dataset

Follows "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1) methodology:
- Section 4: Data design with automatic correctness checks
- Section 3: Shift detection via lexical cues
- Appendix A.1: Scoring by normalized exact match

This dataset creates reasoning traces where models must:
1. Perform step-by-step reasoning
2. Recognize correction signals ("Wait,", "Actually,", "No,")
3. Revise their reasoning appropriately

Unlike the paper's use of large LLMs (o1, DeepSeek-R1), we adapt for small
transformers by creating structured sequences that test the same capabilities.

Reference: https://arxiv.org/html/2601.00514v1#S4
"""

import os
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ShiftType(Enum):
    """Types of reasoning shifts (following paper Section 3.1)"""
    NO_SHIFT = "no_shift"           # Maintain original answer
    SPONTANEOUS = "spontaneous"     # Self-initiated correction
    TRIGGERED = "triggered"         # Externally prompted correction


# Lexical cues for shift detection (Table 10 in paper)
SHIFT_CUES = {
    'reconsideration': ['Wait,', 'Actually,', 'Hold on,', 'Let me reconsider,'],
    'negation': ['No,', 'That\'s wrong,', 'Incorrect,', 'I made a mistake,'],
    'revision': ['The correct answer is', 'I should have', 'Let me redo this,'],
    'uncertainty': ['Hmm,', 'I\'m not sure,', 'Maybe,', 'Perhaps,'],
}


@dataclass
class ReasoningProblem:
    """A single reasoning problem with optional shift"""
    problem: str
    initial_reasoning: str
    shift_cue: Optional[str]
    revised_reasoning: Optional[str]
    initial_answer: int
    correct_answer: int
    shift_type: ShiftType
    
    def to_sequence(self) -> str:
        """Convert to training sequence"""
        if self.shift_type == ShiftType.NO_SHIFT:
            return f"{self.problem}\n{self.initial_reasoning}\nAnswer: {self.correct_answer}"
        else:
            return (f"{self.problem}\n{self.initial_reasoning}\n"
                   f"{self.shift_cue}\n{self.revised_reasoning}\nAnswer: {self.correct_answer}")
    
    def get_segments(self) -> Dict:
        """Get labeled segments for analysis"""
        return {
            'problem': self.problem,
            'pre_shift': self.initial_reasoning,
            'shift_cue': self.shift_cue,
            'post_shift': self.revised_reasoning,
            'initial_answer': self.initial_answer,
            'correct_answer': self.correct_answer,
            'shift_type': self.shift_type.value,
        }


# =============================================================================
# ARITHMETIC PROBLEMS (inspired by paper's Math domain)
# =============================================================================

def generate_arithmetic_problem(
    difficulty: str = 'easy',
    include_shift: bool = False,
    shift_type: ShiftType = ShiftType.TRIGGERED
) -> ReasoningProblem:
    """
    Generate arithmetic problem with step-by-step reasoning.
    
    Following paper Section 4 (Math): "Math word problems test symbolic 
    manipulation and multi-step deduction, with reasoning progress naturally 
    expressed step-by-step."
    """
    
    if difficulty == 'easy':
        # Simple two-step problems
        a, b, c = random.randint(10, 50), random.randint(5, 20), random.randint(2, 10)
        op1 = random.choice(['+', '-'])
        op2 = random.choice(['*', '/'])
        
        if op1 == '+':
            step1_result = a + b
            step1_text = f"First, {a} + {b} = {step1_result}"
        else:
            step1_result = a - b
            step1_text = f"First, {a} - {b} = {step1_result}"
        
        if op2 == '*':
            correct_answer = step1_result * c
            step2_text = f"Then, {step1_result} * {c} = {correct_answer}"
        else:
            correct_answer = step1_result // c
            step2_text = f"Then, {step1_result} / {c} = {correct_answer}"
        
        problem = f"Calculate: ({a} {op1} {b}) {op2} {c}"
        
    elif difficulty == 'medium':
        # Three-step problems
        a, b, c, d = random.randint(5, 30), random.randint(2, 15), random.randint(2, 10), random.randint(2, 5)
        
        step1_result = a * b
        step1_text = f"First, {a} * {b} = {step1_result}"
        
        step2_result = step1_result + c
        step2_text = f"Then, {step1_result} + {c} = {step2_result}"
        
        correct_answer = step2_result // d
        step3_text = f"Finally, {step2_result} / {d} = {correct_answer}"
        
        problem = f"Calculate: (({a} * {b}) + {c}) / {d}"
        step2_text = f"{step1_text}\n{step2_text}\n{step3_text}"
        step1_text = step2_text
        
    else:  # hard
        # Multi-step with potential for error
        a, b, c = random.randint(10, 100), random.randint(5, 50), random.randint(2, 10)
        
        step1_result = a + b
        step1_text = f"Let me think: {a} + {b} = {step1_result}"
        
        correct_answer = step1_result * c - a
        step2_text = f"Then ({step1_result}) * {c} - {a} = {step1_result * c} - {a} = {correct_answer}"
        
        problem = f"Calculate: ({a} + {b}) * {c} - {a}"
        step1_text = f"{step1_text}\n{step2_text}"
    
    if not include_shift:
        return ReasoningProblem(
            problem=problem,
            initial_reasoning=step1_text,
            shift_cue=None,
            revised_reasoning=None,
            initial_answer=correct_answer,
            correct_answer=correct_answer,
            shift_type=ShiftType.NO_SHIFT
        )
    
    # Generate a WRONG initial answer for shift scenarios
    wrong_answer = correct_answer + random.choice([-10, -5, -2, 2, 5, 10])
    wrong_step = step1_text.rsplit('=', 1)[0] + f"= {wrong_answer}"  # Replace final answer
    
    shift_cue = random.choice(SHIFT_CUES['reconsideration'] + SHIFT_CUES['negation'])
    revised_reasoning = f"Let me recalculate.\n{step1_text}"
    
    return ReasoningProblem(
        problem=problem,
        initial_reasoning=wrong_step,
        shift_cue=shift_cue,
        revised_reasoning=revised_reasoning,
        initial_answer=wrong_answer,
        correct_answer=correct_answer,
        shift_type=shift_type
    )


# =============================================================================
# BELIEF REVISION PROBLEMS (analogous to Cryptic Crosswords)
# =============================================================================

def generate_belief_revision_problem(
    include_shift: bool = True
) -> ReasoningProblem:
    """
    Generate problems requiring belief revision.
    
    Analogous to paper's Cryptic Crosswords: "require representational shifts to solve"
    Our version: Problems where initial interpretation is wrong and must be revised.
    """
    
    # Misleading word problems
    scenarios = [
        {
            'problem': "A farmer has 17 sheep. All but 9 run away. How many are left?",
            'wrong_reasoning': "The farmer lost some sheep.\n17 sheep total, some ran away.\nIf all but 9 ran away, then 17 - 9 = 8 ran away.\nSo 17 - 8 = 9 are left... wait, that's circular.",
            'wrong_answer': 8,
            'correct_reasoning': "\"All but 9\" means 9 remain.\nSo the answer is simply 9.",
            'correct_answer': 9,
        },
        {
            'problem': "If you have 3 apples and take away 2, how many do YOU have?",
            'wrong_reasoning': "Starting with 3 apples.\nTake away 2.\n3 - 2 = 1 apple remains.",
            'wrong_answer': 1,
            'correct_reasoning': "The question asks how many YOU have.\nYOU took 2 apples.\nSo YOU have 2.",
            'correct_answer': 2,
        },
        {
            'problem': "A clerk at a butcher shop is 5'10\". What does he weigh?",
            'wrong_reasoning': "This seems like a weight estimation problem.\nAverage weight for 5'10\" is around 160-180 lbs.\nEstimate: 170 lbs.",
            'wrong_answer': 170,
            'correct_reasoning': "A butcher shop clerk weighs MEAT.\nThat's his job - to weigh meat for customers.\nThe answer is: meat (or 0 for numerical).",
            'correct_answer': 0,  # Representing "meat" as 0 for numerical processing
        },
        {
            'problem': "How many times can you subtract 5 from 25?",
            'wrong_reasoning': "25 / 5 = 5.\nSo you can subtract 5 times.\nWait, let me verify: 25-5=20, 20-5=15, 15-5=10, 10-5=5, 5-5=0.\nYes, 5 times.",
            'wrong_answer': 5,
            'correct_reasoning': "You can only subtract 5 from 25 ONCE.\nAfter that, you're subtracting from 20, not 25.\nAnswer: 1",
            'correct_answer': 1,
        },
    ]
    
    scenario = random.choice(scenarios)
    
    if not include_shift:
        # No shift - model should get it right immediately
        return ReasoningProblem(
            problem=scenario['problem'],
            initial_reasoning=scenario['correct_reasoning'],
            shift_cue=None,
            revised_reasoning=None,
            initial_answer=scenario['correct_answer'],
            correct_answer=scenario['correct_answer'],
            shift_type=ShiftType.NO_SHIFT
        )
    
    shift_cue = random.choice(SHIFT_CUES['reconsideration'])
    
    return ReasoningProblem(
        problem=scenario['problem'],
        initial_reasoning=scenario['wrong_reasoning'],
        shift_cue=shift_cue,
        revised_reasoning=scenario['correct_reasoning'],
        initial_answer=scenario['wrong_answer'],
        correct_answer=scenario['correct_answer'],
        shift_type=ShiftType.TRIGGERED
    )


# =============================================================================
# SEQUENCE CONTINUATION (analogous to Rush Hour - spatial/sequential)
# =============================================================================

def generate_sequence_problem(
    include_shift: bool = True
) -> ReasoningProblem:
    """
    Generate sequence continuation problems.
    
    Analogous to paper's Rush Hour: requires understanding sequential patterns.
    """
    
    # Different sequence types
    sequence_types = [
        {
            'name': 'arithmetic',
            'sequence': lambda start, diff: [start + i * diff for i in range(5)],
            'make_wrong': lambda seq: seq[-1] + random.choice([1, -1, 2]),
        },
        {
            'name': 'geometric',
            'sequence': lambda start, ratio: [start * (ratio ** i) for i in range(5)],
            'make_wrong': lambda seq: seq[-1] + random.choice([1, -1, seq[-1] // 2]),
        },
        {
            'name': 'fibonacci_like',
            'sequence': lambda a, b: [a, b, a+b, b+(a+b), (a+b)+(b+(a+b))],
            'make_wrong': lambda seq: seq[-1] + random.choice([1, -1]),
        },
    ]
    
    seq_type = random.choice(sequence_types)
    
    if seq_type['name'] == 'arithmetic':
        start, diff = random.randint(1, 10), random.randint(2, 5)
        sequence = seq_type['sequence'](start, diff)
        pattern_desc = f"arithmetic sequence with difference {diff}"
    elif seq_type['name'] == 'geometric':
        start, ratio = random.randint(2, 4), 2
        sequence = seq_type['sequence'](start, ratio)
        pattern_desc = f"geometric sequence with ratio {ratio}"
    else:
        a, b = random.randint(1, 5), random.randint(1, 5)
        sequence = seq_type['sequence'](a, b)
        pattern_desc = "Fibonacci-like sequence (each term = sum of previous two)"
    
    display_seq = sequence[:4]
    correct_answer = sequence[4]
    wrong_answer = seq_type['make_wrong'](sequence)
    
    problem = f"What comes next: {', '.join(map(str, display_seq))}, ?"
    
    wrong_reasoning = f"Looking at the sequence: {', '.join(map(str, display_seq))}\n"
    wrong_reasoning += f"Let me guess the pattern...\nMaybe each number increases by varying amounts.\n"
    wrong_reasoning += f"Next number might be {wrong_answer}."
    
    correct_reasoning = f"This is a {pattern_desc}.\n"
    correct_reasoning += f"Pattern: {', '.join(map(str, display_seq))}\n"
    correct_reasoning += f"Next: {correct_answer}"
    
    if not include_shift:
        return ReasoningProblem(
            problem=problem,
            initial_reasoning=correct_reasoning,
            shift_cue=None,
            revised_reasoning=None,
            initial_answer=correct_answer,
            correct_answer=correct_answer,
            shift_type=ShiftType.NO_SHIFT
        )
    
    shift_cue = random.choice(SHIFT_CUES['revision'])
    
    return ReasoningProblem(
        problem=problem,
        initial_reasoning=wrong_reasoning,
        shift_cue=shift_cue,
        revised_reasoning=correct_reasoning,
        initial_answer=wrong_answer,
        correct_answer=correct_answer,
        shift_type=ShiftType.TRIGGERED
    )


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_reasoning_dataset(
    n_samples: int = 1000,
    shift_ratio: float = 0.5,  # Following paper: ~50% have shifts for training
    problem_types: List[str] = ['arithmetic', 'belief', 'sequence'],
    output_dir: str = 'data/reasoning_shift',
    seed: int = 42
) -> Dict:
    """
    Generate complete reasoning shift dataset.
    
    Returns:
        Dict with train/val splits and metadata
    """
    random.seed(seed)
    np.random.seed(seed)
    
    problems = []
    
    generators = {
        'arithmetic': lambda shift: generate_arithmetic_problem(
            difficulty=random.choice(['easy', 'medium']),
            include_shift=shift,
            shift_type=ShiftType.TRIGGERED if shift else ShiftType.NO_SHIFT
        ),
        'belief': lambda shift: generate_belief_revision_problem(include_shift=shift),
        'sequence': lambda shift: generate_sequence_problem(include_shift=shift),
    }
    
    for i in range(n_samples):
        include_shift = random.random() < shift_ratio
        problem_type = random.choice(problem_types)
        
        problem = generators[problem_type](include_shift)
        problems.append({
            'id': i,
            'type': problem_type,
            'sequence': problem.to_sequence(),
            'segments': problem.get_segments(),
            'has_shift': include_shift,
        })
    
    # Split into train/val
    random.shuffle(problems)
    split_idx = int(0.8 * len(problems))
    train_problems = problems[:split_idx]
    val_problems = problems[split_idx:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as text files for tokenization
    train_text = '\n\n---\n\n'.join([p['sequence'] for p in train_problems])
    val_text = '\n\n---\n\n'.join([p['sequence'] for p in val_problems])
    
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write(train_text)
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write(val_text)
    
    # Save metadata
    metadata = {
        'n_train': len(train_problems),
        'n_val': len(val_problems),
        'shift_ratio': shift_ratio,
        'problem_types': problem_types,
        'train_shift_count': sum(1 for p in train_problems if p['has_shift']),
        'val_shift_count': sum(1 for p in val_problems if p['has_shift']),
    }
    
    np.save(os.path.join(output_dir, 'train_problems.npy'), train_problems, allow_pickle=True)
    np.save(os.path.join(output_dir, 'val_problems.npy'), val_problems, allow_pickle=True)
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata, allow_pickle=True)
    
    print(f"Generated {len(train_problems)} train, {len(val_problems)} val problems")
    print(f"Shift ratio - Train: {metadata['train_shift_count']/len(train_problems):.2%}, "
          f"Val: {metadata['val_shift_count']/len(val_problems):.2%}")
    
    return {
        'train': train_problems,
        'val': val_problems,
        'metadata': metadata,
    }


# =============================================================================
# SHIFT DETECTION (Following paper Table 10)
# =============================================================================

def detect_shifts(text: str) -> Dict:
    """
    Detect reasoning shifts using lexical cues (paper Table 10).
    
    Returns:
        Dict with shift detection results
    """
    shifts_found = []
    
    for category, cues in SHIFT_CUES.items():
        for cue in cues:
            if cue.lower() in text.lower():
                pos = text.lower().find(cue.lower())
                shifts_found.append({
                    'category': category,
                    'cue': cue,
                    'position': pos,
                    'context': text[max(0, pos-20):pos+len(cue)+20]
                })
    
    return {
        'has_shift': len(shifts_found) > 0,
        'shift_count': len(shifts_found),
        'shifts': shifts_found,
    }


def compute_paper_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> Dict:
    """
    Compute metrics following paper Section 3.
    
    Metrics:
    - P(S): Shift prevalence
    - P(✓|S=1): Accuracy given shift
    - P(✓|S=0): Accuracy given no shift
    - Entropy at shift points
    """
    
    total = len(predictions)
    shift_count = sum(1 for p in predictions if p.get('detected_shift', False))
    
    # Separate by shift status
    shift_correct = 0
    shift_total = 0
    no_shift_correct = 0
    no_shift_total = 0
    
    for pred, gt in zip(predictions, ground_truth):
        correct = (pred.get('answer') == gt.get('correct_answer'))
        has_shift = pred.get('detected_shift', False)
        
        if has_shift:
            shift_total += 1
            if correct:
                shift_correct += 1
        else:
            no_shift_total += 1
            if correct:
                no_shift_correct += 1
    
    metrics = {
        'P(S)': shift_count / total if total > 0 else 0,
        'P(correct|S=1)': shift_correct / shift_total if shift_total > 0 else 0,
        'P(correct|S=0)': no_shift_correct / no_shift_total if no_shift_total > 0 else 0,
        'total_samples': total,
        'shift_count': shift_count,
        'no_shift_count': no_shift_total,
    }
    
    # Paper finding: P(correct|S=0) > P(correct|S=1) indicates shifts are harmful
    if metrics['P(correct|S=0)'] > 0:
        metrics['shift_benefit'] = metrics['P(correct|S=1)'] - metrics['P(correct|S=0)']
    else:
        metrics['shift_benefit'] = 0
    
    return metrics


# =============================================================================
# TOKENIZED DATASET FOR TRAINING
# =============================================================================

def create_tokenized_dataset(
    problems: List[Dict],
    tokenizer,
    max_length: int = 256,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Convert problems to tokenized tensors for training.
    
    Returns:
        input_ids: Token indices [n_samples, max_length]
        labels: Target token indices [n_samples, max_length]
        metadata: Problem metadata for analysis
    """
    all_input_ids = []
    all_labels = []
    metadata = []
    
    for prob in problems:
        # Tokenize sequence
        tokens = tokenizer.encode(prob['sequence'])
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif len(tokens) < max_length:
            tokens = tokens + [tokenizer.pad_token_id or 0] * (max_length - len(tokens))
        
        # For language modeling: labels are input shifted by 1
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        metadata.append(prob['segments'])
    
    return (
        torch.tensor(all_input_ids, device=device),
        torch.tensor(all_labels, device=device),
        metadata
    )


if __name__ == '__main__':
    # Generate dataset
    data = generate_reasoning_dataset(
        n_samples=2000,
        shift_ratio=0.5,
        output_dir='data/reasoning_shift'
    )
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE PROBLEMS")
    print("="*60)
    
    for i, prob in enumerate(data['train'][:3]):
        print(f"\n--- Problem {i+1} ({prob['type']}) ---")
        print(f"Has shift: {prob['has_shift']}")
        print(prob['sequence'])
        
        # Test shift detection
        detection = detect_shifts(prob['sequence'])
        print(f"Shift detection: {detection['has_shift']} ({detection['shift_count']} cues found)")
