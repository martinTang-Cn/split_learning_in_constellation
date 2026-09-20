"""Run orbit generation followed by paired multimodal CROMA training."""

from generate_orbit_data import main as generate_orbit_data
from multimodal_croma_demo import main as run_multimodal_training


if __name__ == "__main__":
    generate_orbit_data()
    run_multimodal_training()
