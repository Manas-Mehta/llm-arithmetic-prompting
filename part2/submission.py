import json
import collections
import argparse
import random
import numpy as np
import requests
import re

def your_netid():
    YOUR_NET_ID = 'mm14444'
    return YOUR_NET_ID

def your_hf_token():
    YOUR_HF_TOKEN = 'YOUR_HF_TOKEN'
    return YOUR_HF_TOKEN


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    # Examples use reversed digits so model adds left-to-right (ones place first)
    # This aligns carry propagation with autoregressive generation direction
    prefix = (
        "Question: What is 7654321+7654321?\nAnswer: 4319642\n"
        "Question: What is 1111111+1111111?\nAnswer: 2222222\n"
        "Question: What is 8765432+9876543?\nAnswer: 7642085\n"
        "Question: What is 1234567+7654321?\nAnswer: 8888888\n"
        "Question: What is 3333333+4444444?\nAnswer: 7777777\n"
        "Question: What is 0000005+0000005?\nAnswer: 00000001\n"
        "Question: What is "
    )
    suffix = "?\nAnswer: "

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys.
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 50,
        'temperature': 0.01,
        'top_k': 1,
        'top_p': 0.1,
        'repetition_penalty': 1,
        'stop': []}

    return config


def your_pre_processing(s):
    # Reverse digits of each number so model generates ones digit first
    parts = s.split('+')  # string split, not arithmetic
    return f"{parts[0][::-1]}+{parts[1][::-1]}"  # string reversal, not arithmetic


def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    # Extract first number and reverse digits back to normal order
    match = re.search(r'\d+', output_string)
    if match:
        return int(match.group()[::-1])  # string reversal, not arithmetic
    return 0
