"""_summary_

Returns:
    _type_: _description_
"""

from PIL import Image
from music21 import stream
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.utils.vis_utils import plot_model


def plot_sheet_music(melody: list) -> Image.Image:
    """_summary_

    Args:
        melody (list): _description_. Defaults to None.
    """
    melody_stream = stream.Stream(melody)

    return Image.open(str(melody_stream.write("musicxml.png")))

def plot_chords(count_dict: dict) -> None:
    """_summary_

    Args:
        count_dict (dict): Dictionary with the note count in a music21.stream

    Returns:
        None.
        Opens a plot window.
    """
    plt.figure()
    plt.bar(x= count_dict.keys(), height=count_dict.values())
    plt.show()


def plot_learning_curve(model_history):
        dataframe = pd.DataFrame(model_history.history)
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Learning Curve')
        pl = sns.lineplot(data=dataframe['val_accuracy'])

        return fig


def plot_model_graph(model, save_2_filename: str, shapes: bool=True, layer_names: bool=True):

    return plot_model(model=model, to_file=save_2_filename, show_shapes=shapes, show_layer_names=layer_names)
