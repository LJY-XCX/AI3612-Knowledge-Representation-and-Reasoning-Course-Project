from collections import defaultdict
import json
from re import L
from tkinter import E
import os
import random
import torch

from copy import deepcopy
from typing import List
from transformers import MarianMTModel, MarianTokenizer
from transformers import MarianTokenizer, MarianMTModel

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from tqdm import tqdm

import random
import synonyms


def get_synonyms(word):
    candidates = list()
    for synonym in synonyms.nearby(word)[0]:
        candidates.append(synonym)
    if word in candidates:
        candidates.remove(word)
    for candidate in candidates:
        if len(candidate) == len(word):
            return candidate
    return word
