import numpy as np
import plotly.graph_objects as go
from numpy.linalg import norm


def primitive_vector(v):
    """Reduce vector to primitive integer direction by dividing by gcd."""
    g = np.gcd.reduce(np.abs(v))
    return tuple(v // g) if g != 0 else tuple(v)


def generate_diophantine_directions(max_wavelength=60, single_wavelength=False):
    """Generate unique primitive directions and count commensurate multiples."""
    counts = {}

    if not single_wavelength:
        N = int(max_wavelength / 3)
        for x in range(-N, N + 1):
            for y in range(-N, N + 1):
                for z in range(-N, N + 1):
                    if x == y == z == 0:
                        continue
                    pv = primitive_vector(np.array([x, y, z]))
                    counts[pv] = counts.get(pv, 0) + 1
    else:
        L = max_wavelength
        for x in range(-L, L + 1):
            for y in range(-L, L + 1):
                for z in range(-L, L + 1):
                    if abs(x) + abs(y) + abs(z) == L:
                        pv = primitive_vector(np.array([x, y, z]))
                        counts[pv] = counts.get(pv, 0) + 1

    return counts


def make_point_cloud(counts):
    """Normalize primitive directions and prepare color data."""
    dirs = np.array(list(counts.keys()), dtype=float)
    mags = np.linalg.norm(dirs, axis=1)
    unit_dirs = dirs / mags[:, None]
    weights = np.array(list(counts.values()), dtype=float)
    return unit_dirs, weights


def plot_diophantine_sphere(
    max_wavelength=60, filter_uniques=False, single_wavelenth=False, radius=10.0
):
    counts = generate_diophantine_directions(max_wavelength, single_wavelenth)
    if filter_uniques:
        filtered = {}
        for key, value in counts.items():
            if value > 1:
                filtered[key] = value
        points, weights = make_point_cloud(filtered)
    else:
        points, weights = make_point_cloud(counts)

    # Normalize weights for color mapping
    colors = np.log1p(weights)

    # Outline sphere (for context)
    phi, theta = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
    x_s = np.cos(phi) * np.sin(theta)
    y_s = np.sin(phi) * np.sin(theta)
    z_s = np.cos(theta)

    fig = go.Figure()

    # Add faint wireframe sphere
    fig.add_trace(
        go.Scatter3d(
            x=x_s.flatten(),
            y=y_s.flatten(),
            z=z_s.flatten(),
            mode="markers",
            marker=dict(size=1, color="lightgray", opacity=0.1),
            showlegend=False,
        )
    )

    # Normalize colors and opacities together
    weights = np.array(list(counts.values()), dtype=float)
    colors = np.log1p(weights)
    dp = np.column_stack([weights * 0.01, weights * 0.01, weights * 0.01])
    # points += dp
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=radius,
                color=colors,
                colorscale="Viridis",
                colorbar=dict(title="Count"),
                opacity=0.5,
            ),
            text=[f"count={int(c)}" for c in weights],
            name="",
        )
    )

    fig.update_layout(
        title=f"Diophantine Commensurability Sphere (N={max_wavelength})",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        template="simple_white",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.show()


# plot_diophantine_sphere(max_wavelength=20, filter_uniques=False, single_wavelenth=True, radius=5)
plot_diophantine_sphere(
    max_wavelength=70, filter_uniques=False, single_wavelenth=False, radius=2
)
