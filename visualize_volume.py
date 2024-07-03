import numpy as np
import plotly.graph_objects as go

def display_3d_model(volume):

    x, y, z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        opacity=0.1 , 
        surface_count=10,  # Number of isosurfaces
        isomin=volume.min(),
        isomax=volume.max(),
        colorscale='Viridis'  
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ))

    fig.show()

if __name__ == "__main__":
    file_path = "./10_512.npy"
    volume = np.load(file_path)
    volume = np.maximum(volume, 0)

    display_3d_model(volume)