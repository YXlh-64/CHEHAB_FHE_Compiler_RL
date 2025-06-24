from torchviz import make_dot
from embeddings_AETR_CLS import TreeAutoencoder ,config
from torchview import draw_graph


model = TreeAutoencoder()

graph = draw_graph(
    model,
    input_size=(1, 32),          # batch, seq_len
    device="meta",               # no real memory used
    expand_nested=True,          # show encoder & decoder blocks
    hide_inner_tensors=True,     # ← this is the one you want
    # hide_module_functions=True is already the default
    depth=3,                     # tweak if you want even less detail
    graph_attr={"rankdir": "LR"} # left-to-right flow
)

graph.visual_graph.render(
    "tree_autoencoder_modules",
    format="png",
    cleanup=True
)
