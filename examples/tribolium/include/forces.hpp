#include "../../../include/kocs.hpp"

using namespace kocs;

enum class CellType {
  Serosa,
  Anchor,
  Pole,
  Embryo
};

template<typename SimulationConfig>
struct SystemForces {
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

  SystemForces(
    View<Vector> grid_view_,
    View<Vector> normals_view_
  ) : grid_view(grid_view_)
    , normals_view(normals_view_)
    , grid(grid_view_) {
      grid.rebuild();
    }

  View<Vector> grid_view;
  View<Vector> normals_view;
  acceleration::Grid<Vector> grid;

  GENERIC_FORCE_OP(GENERIC_REF(Vector, position)) {
    int nearest_grid_point_idx = grid.get_nearest_point_index(position.self);
    if (nearest_grid_point_idx >= 0) {
      Vector relative_position = position.self - grid_view(nearest_grid_point_idx);
      Vector normal = normals_view(nearest_grid_point_idx);

      Scalar F = -12.0f * relative_position.dot(normal);
      position.delta += F * normal;
    }
  }
};



template<typename SimulationConfig>
struct ForceBetweenCellsCustom {
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

  ForceBetweenCellsCustom(View<int> cell_types_view_) : cell_types_view(cell_types_view_) { }

  View<int> cell_types_view;

  PAIRWISE_FORCE_OP(PAIRWISE_REF(Vector, position)) {
    const CellType this_type = static_cast<CellType>(cell_types_view(i));
    const CellType other_type = static_cast<CellType>(cell_types_view(j));

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      position.delta += (Kokkos::exp(-distance) - 0.5f) * displacement;
    }
    else {
      if (other_type == CellType::Pole)
        position.delta += (-Kokkos::exp(-distance)) * displacement;
      position.delta += (Kokkos::exp(-distance) - 0.4f) * displacement;
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      friction += 1.0;
  }
};

template<typename SimulationConfig>
struct ForceBetweenCellsMorse {
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

  ForceBetweenCellsMorse(View<int> cell_types_view_) : cell_types_view(cell_types_view_) { }

  View<int> cell_types_view;

  PAIRWISE_FORCE_OP(PAIRWISE_REF(Vector, position)) {
    const CellType this_type = static_cast<CellType>(cell_types_view(i));
    const CellType other_type = static_cast<CellType>(cell_types_view(j));

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      position.delta += forces::Morse(displacement, distance, 1.0f, 2.0f);
    }
    else {
      if (other_type == CellType::Pole)
        position.delta += position.delta += forces::Morse(displacement, distance, 1.0f, 2.0f);
      position.delta += position.delta += forces::Morse(displacement, distance, 1.0f, 2.0f);
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      friction += 1.0;
  }
};
