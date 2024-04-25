
def reverse_sort_dict(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

def test_reverse_sort_dict():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = reverse_sort_dict(d)
    assert result == {'c': 3, 'b': 2, 'a': 1}
    print(result)
    print("okay")

test_reverse_sort_dict()