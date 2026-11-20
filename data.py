
import numpy as np

LAYER_IDS = [1, 2, 3, 4, 7, 9, 5, 6, 8]
LAYER_TO_INDEX = {layer: i for i, layer in enumerate(LAYER_IDS)}


def read_combinations(filename):
    features_list = []
    masks_list = []
    labels = []

    with open(filename, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            values = np.asarray([float(v) for v in line.split(",")], dtype=np.float32)

            # Last value is the real/fake label
            label_value = values[-1]

            if label_value not in (0, 1):
                raise ValueError(f"Invalid label at line {line_num}: {label_value}")

            cross_values = values[:-1]

            if len(cross_values) % 4 != 0:
                raise ValueError(f"Invalid number of cross features at line {line_num}: {len(cross_values)}")

            crosses = cross_values.reshape(-1, 4)

            X = np.zeros((9, 3), dtype=np.float32)
            valid_mask = np.zeros(9, dtype=bool)

            for cross in crosses:
                v1, v2, v3, layer_value = cross

                layer = int(round(layer_value))

                if not np.isclose(layer_value, layer):
                    raise ValueError(f"Non-integer layer at line {line_num}: {layer_value}")

                if layer not in LAYER_TO_INDEX:
                    raise ValueError(f"Unknown layer at line {line_num}: {layer}")

                token = LAYER_TO_INDEX[layer]

                if valid_mask[token]:
                    raise ValueError(f"Duplicate layer {layer} at line {line_num}")

                if layer in [1, 2, 3]:
                    # SVT
                    X[token] = [v1, v2, v3]

                elif layer in [4, 7, 9]:
                    # BMTC
                    X[token] = [v1, v2, 0.0]

                elif layer in [5, 6, 8]:
                    # BMTZ: encode phi using sin and cos
                    X[token] = [v1, np.sin(v2), np.cos(v2)]

                valid_mask[token] = True

            if not valid_mask.any():
                continue

            features_list.append(X)
            masks_list.append(valid_mask)
            labels.append(int(label_value))

    # Explicit shapes also handle an empty dataset
    features = np.asarray(features_list, dtype=np.float32).reshape(-1, 9, 3)
    masks = np.asarray(masks_list, dtype=bool).reshape(-1, 9)
    labels = np.asarray(labels, dtype=np.int64)

    return features, masks, labels

# --------------------------------------------------
# Example usage
# --------------------------------------------------

if __name__ == "__main__":

    features, valid_mask, labels = (
        read_combinations("realFakeCrossCombos.csv")
    )

    print("Number of tracks:", len(features))

    print("Features shape:", features.shape)
    print("Mask shape:", valid_mask.shape)
    print("States shape:", labels.shape)

    print("\nFirst track features:")
    print(features[0])

    print("\nFirst track valid mask:")
    print(valid_mask[0])

    print("\nFirst track target:")
    print(labels[0])