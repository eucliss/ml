import itertools

sequence = itertools.accumulate(itertools.cycle(map(ord, "Close")))
print(list(itertools.islice(sequence, 10)))