# ruff: noqa: F722
from __future__ import annotations

import enum
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import einops

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from jaxtyping import Float, Inexact, Array


class AxisState(enum.Enum):
    """Enumeration of axis orderings for 3D data."""
    ZYX = "zyx"
    ZXY = "zxy"
    YXZ = "yxz"
    YZX = "yzx"
    XZY = "xzy"
    XYZ = "xyz"


def generate_einops_pattern(
    target_state: AxisState
) -> str:
    """
    Generate einops rearrangement pattern from ZYX to target axis order.
    """
    source_state: AxisState = AxisState.ZYX
    expanded_source_state = " ".join(source_state.value)
    expanded_target_state = " ".join(target_state.value)
    # d is the 3D vector field component axis
    pattern = f"{expanded_source_state} d -> {expanded_target_state} d"
    return pattern


def get_vectorfield_component_indices(
    axis_state: AxisState,
) -> tuple[int, int]:
    """
    Select the in-plane vectorfield components according to the AxisState.
    """
    axis_to_index = {'z': 0, 'y': 1, 'x': 2}
    return tuple(axis_to_index[axis] for axis in axis_state.value[1:])


def _make_axis_dropdown():
    """
    Axis order dropdown.
    Initial state is assumed as ZYX (depth, height, width).
    """
    initial_axis_state = AxisState.ZYX
    return widgets.Dropdown(
        options=[axis_state.value for axis_state in AxisState],
        value=initial_axis_state.value,
        description='Axis Order:',
    )


def _make_subsample_factor_slider() -> widgets.IntSlider:
    """Subsample factor slider for in-plane quiver visualization."""
    return widgets.IntSlider(
        value=2,
        min=1,
        max=20,
        step=1,
        description="Subsample Factor:",
        continuous_update=False,
    )


def _make_axis_index_slider(
    axis_size: int,
) -> widgets.IntSlider:
    """Axis index slider for selecting slice in quiver visualization."""
    return widgets.IntSlider(
        value=axis_size // 2,
        min=0,
        max=axis_size - 1,
        step=1,
        description="Slice Index:",
        continuous_update=True,
    )


def _color_map_selection_dropdown() -> widgets.Dropdown:
    choices: list[str] = [
        'none',
        'viridis',
        'plasma',
        'inferno',
        'magma',
        'cividis',
    ]
    return widgets.Dropdown(
        options=choices,
        value='viridis',
        description='Color Map:',
    )


class QuiverVisualizer:
    """
    Visualize 3D vector fields using quiver plots.
    """
    def __init__(
        self,
        ax: Axes,
        vector_field: Inexact[Array, "d h w 3"],
        axis_index_slider: widgets.IntSlider,
        subsample_factor_slider: widgets.IntSlider,
        axis_order_dropdown: widgets.Dropdown,
        color_map_dropdown: widgets.Dropdown,
    ) -> None:
        """
        Initialize QuiverVisualizer with a 3D vector field.

        Parameters
        ----------
        vector_field : Inexact[Array, "d h w 3"]
            3D vector field to visualize.
        """
        # this is the basic input vector field in d h w 3 format
        self._input_vector_field = vector_field
        # ui controls
        self._axis_index_slider = axis_index_slider
        self._subsample_factor_slider = subsample_factor_slider
        self._axis_order_dropdown = axis_order_dropdown
        self._color_map_dropdown = color_map_dropdown

        self._ax = ax

        # Stateful information about current display vector field.
        # The display vector field can be different from the input vector field
        # due to axis rearrangements/permutations and in-plane subsampling.
        self._display_vector_field: Inexact[Array, "h w 3"] | None = None

        self._ui = widgets.HBox(
            [self._axis_index_slider,
             self._subsample_factor_slider,
             self._axis_order_dropdown,
             self._color_map_dropdown]
        )
        self.ux = widgets.VBox([self.figure.canvas, self._ui])


    @classmethod
    def create(
        cls,
        vector_field: Inexact[Array, "d h w 3"],
        ax: Axes | None = None,
    ) -> QuiverVisualizer:
        """
        Factory method to create QuiverVisualizer with default UI components.

        Parameters
        ----------
        vector_field : Inexact[Array, "d h w 3"]
            3D vector field to visualize.
        
        Returns
        -------
        QuiverVisualizer
            Initialized QuiverVisualizer instance.
        """
        if ax is None:
            with plt.ioff():
                fig, ax = plt.subplots()

        axis_order_dropdown = _make_axis_dropdown()
        subsample_factor_slider = _make_subsample_factor_slider()
        axis_index_slider = _make_axis_index_slider(vector_field.shape[0])
        color_map_dropdown = _color_map_selection_dropdown()

        visualizer = cls(
            ax=ax,
            vector_field=vector_field,
            axis_index_slider=axis_index_slider,
            subsample_factor_slider=subsample_factor_slider,
            axis_order_dropdown=axis_order_dropdown,
            color_map_dropdown=color_map_dropdown,
        )
        visualizer._wire_all()
        visualizer.update_display_vector_field()
        visualizer._draw_quiver()
        return visualizer


    def query_vector_field(self) -> Inexact[Array, "d h w 3"]:
        """
        Query the vector field in the given axis state and subsample factor.
        """
        target_axis_state = AxisState(self._axis_order_dropdown.value)
        einops_pattern = generate_einops_pattern(target_axis_state)
        subsample_index = self.query_subsample_index_tuple()
        # permute axes according to target axis state
        rearranged_field = einops.rearrange(
            self._input_vector_field,
            einops_pattern
        )
        # only select subsampled information for quiver plotting
        # inside the visualization planes
        subsampled_field: Inexact[Array, "d h w 3"] = rearranged_field[:, *subsample_index, :]
        return subsampled_field
        
        
    def query_subsample_index_tuple(self) -> tuple[slice, slice]:
        """
        Get the subsample index tuple that returns a subsampled 2D plane
        from the 3D vector field based on the current subsample factor.
        """
        subsample_factor: int = self._subsample_factor_slider.value
        plane_slice = tuple((
            slice(None, None, subsample_factor),
            slice(None, None, subsample_factor),
            )
        )
        return plane_slice
    

    def update_display_vector_field(self) -> None:
        """
        Update the display vector field based on current UI settings.
        """
        vector_field = self.query_vector_field()
        self._display_vector_field = vector_field

    
    def get_display_vector_field(self) -> Inexact[Array, "h w 2"]:
        """
        Get the 2D vector field slice for quiver plotting based on current UI settings
        and the stored 3D display vector field.
        """
        axis_state = AxisState(self._axis_order_dropdown.value)
        axis_index = self._axis_index_slider.value
        subsample_index = self.query_subsample_index_tuple()
        vector_field = self._display_vector_field
        components  = get_vectorfield_component_indices(axis_state)
        field_slice = np.take(
            vector_field[axis_index, *subsample_index],
            components,
            axis=-1
        )
        return field_slice
    

    def get_quiver_optargs(self) -> tuple[tuple, dict]:
        """
        Get additional quiver plot arguments based on current UI settings.
        """
        color_map_name = self._color_map_dropdown.value
        if color_map_name != 'none':
            C = np.linalg.norm(
                self.get_display_vector_field(),
                axis=-1
            )
            return ((C, ), {'cmap': color_map_name})
        return ((), {})
    
    
    def _draw_quiver(self) -> None:
        vector_field_2D = self.get_display_vector_field()
        h, w, _ = vector_field_2D.shape
        Y, X = np.mgrid[0:h, 0:w]
        U = vector_field_2D[..., 0]
        V = vector_field_2D[..., 1]

        self._ax.clear()
        optargs, optkwargs = self.get_quiver_optargs()
        self._ax.quiver(X, Y, U, V, *optargs, **optkwargs)
        self._ax.set_aspect('equal')
        self.figure.canvas.draw_idle()


    def _on_axis_order_change(self, change) -> None:
        # make new 3D display vector field (permute + subsample)
        display_vector_field = self.query_vector_field()
        self._display_vector_field = display_vector_field

        # update axis index slider (axes size may have changed)
        new_max_index = display_vector_field.shape[0] - 1
        # update axis index slider max value
        self._axis_index_slider.max = new_max_index
        # try to keep the same index if possible
        self._axis_index_slider.value = min(
            self._axis_index_slider.value,
            new_max_index
        )
        self._draw_quiver()


    def _on_subsample_factor_change(self, change) -> None:
        # make new 3D display vector field (permute + subsample)
        display_vector_field = self.query_vector_field()
        self._display_vector_field = display_vector_field
        # redraw quiver
        self._draw_quiver()


    def _on_axis_index_change(self, change) -> None:
        # redraw quiver
        self._draw_quiver()


    def _on_color_map_change(self, change) -> None:
        # redraw quiver
        self._draw_quiver()


    def _wire_all(self) -> None:
        self._axis_order_dropdown.observe(
            self._on_axis_order_change,
            names='value'
        )
        self._subsample_factor_slider.observe(
            self._on_subsample_factor_change,
            names='value'
        )
        self._axis_index_slider.observe(
            self._on_axis_index_change,
            names='value'
        )
        self._color_map_dropdown.observe(
            self._on_color_map_change,
            names='value'
        )


    def _make_axis_index_slider(self) -> widgets.IntSlider:
        # we quey the vector field in dependence of other UI settings

        vector_field = self.query_vector_field()
        max_index = vector_field.shape[0] - 1
        
        if self._axis_index_slider is not None:
            value = min(self._axis_index_slider.value, max_index)
        else:
            value = max_index // 2

        return widgets.IntSlider(
            value=value,
            min=0,
            max=max_index,
            step=1,
            description="Slice Index:",
            continuous_update=True,
        )


    @property
    def figure(self) -> Figure:
        """Get the matplotlib figure associated with the visualizer."""
        return self._ax.get_figure()


    @staticmethod
    def _make_subsample_factor_slider() -> widgets.IntSlider:
        return widgets.IntSlider(
            value=2,
            min=1,
            max=20,
            step=1,
            description="Subsample Factor:",
            continuous_update=False,
        )