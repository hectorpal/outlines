import time

import torch
from dotenv import load_dotenv

import outlines

# prompt: set variable 'my_device' to the available cuda


def t1(model):
    prompt = """You are a sentiment-labelling assistant.
    Is the following review positive or negative?

    Review: This restaurant is just awesome!
    """

    generator = outlines.generate.choice(model, ["Positive", "Negative"])
    answer = generator(prompt)
    print(answer)


def t2(model):
    prompt = "<s>result of 9 + 9 = 18</s><s>result of 1 + 2 = "
    answer = outlines.generate.format(model, int)(prompt)
    print(answer)


def t3(model):
    prompt = "sqrt(2)="
    generator = outlines.generate.format(model, float)
    answer = generator(prompt, max_tokens=10)
    print(answer)


def t4u(model):
    prompt = "What is the IP address of the Google DNS servers? "

    start_time = time.time()

    generator = outlines.generate.text(model)
    unstructured = generator(prompt, max_tokens=30)

    execution_time = time.time() - start_time

    # What is the IP address of the Google DNS servers?
    #
    # Passive DNS servers are at DNS servers that are private.
    # In other words, both IP servers are private. The database
    # does not contain Chelsea Manning

    print(unstructured)
    print(f"Execution time: {execution_time} seconds")


def t4s(model):
    prompt = "What is the IP address of the Google DNS servers? "

    start_time = time.time()

    pattern = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    generator = outlines.generate.regex(
        model,
        pattern,
    )
    structured = generator(prompt, max_tokens=30)

    execution_time = time.time() - start_time
    print(structured)
    # What is the IP address of the Google DNS servers?
    # 2.2.6.1

    print(f"Execution time {pattern}: {execution_time} seconds")


def t5(model):
    prompt = "What is the IP address of the Google DNS servers? "
    start_time = time.time()

    pattern = r"((\d){1,3}\.){3}(\d){1,3}"

    generator = outlines.generate.regex(
        model,
        pattern,
    )
    structured = generator(prompt, max_tokens=30)

    execution_time = time.time() - start_time
    print(structured)
    # What is the IP address of the Google DNS servers?
    # 2.2.6.1

    print(f"Execution time {pattern}: {execution_time} seconds")


def t6(model):
    prompt = "What is the IP address of the Google DNS servers? "
    start_time = time.time()

    pattern = r"((\d){1,3}\.)+(\d){1,3}"
    generator = outlines.generate.regex(
        model,
        pattern,
    )
    structured = generator(prompt, max_tokens=30)

    execution_time = time.time() - start_time
    print(structured)
    # What is the IP address of the Google DNS servers?
    # 2.2.6.1

    print(f"Execution time {pattern}: {execution_time} seconds")


if __name__ == "__main__":
    load_dotenv()
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {my_device}")
    print()

    # model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"

    # might be better
    model = outlines.models.transformers(model_name, device=my_device)

    t1(model)
    print()
    t2(model)
    print()
    try:
        t3(model)  # Sometimes fail because result is not float
    except ValueError:
        pass
    print()
    t4u(model)
    print()
    t4s(model)
    print()
    t5(model)
    print()
    t6(model)
