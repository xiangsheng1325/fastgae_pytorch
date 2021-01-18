# dataset brief description
Google dataset contains 875k nodes, we stored it as a binary file.
## how to open dataset
in bash:
```bash
tar -zxvf preprocessed_google.tar.gz
```
and in python:
```python
import pickle
google_graph = pickle.load(open("google.graph", "rb"))
print(google_graph.number_of_nodes())
print(google_graph.number_of_edges())
```

