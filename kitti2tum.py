#!/usr/bin/env python3
"""
Convert KITTI format trajectory to TUM format.
KITTI: 4x4 transformation matrix as 16 values [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1]
TUM: timestamp x y z q_x q_y q_z q_w
"""

import numpy as np
import sys

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion [qx, qy, qz, qw].
    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qx, qy, qz, qw])

def transform_trajectory_to_origin(matrices):
    """
    Transform a list of 4x4 matrices so the first frame becomes the origin.
    Returns transformed matrices where first matrix becomes identity.
    """
    if not matrices:
        return []
    
    # Get inverse of first matrix
    first_matrix_inv = transform_original_extrinsic_to_inverse(matrices[0])
    
    # Transform all matrices
    transformed_matrices = []
    for matrix in matrices:
        transformed_matrix = first_matrix_inv @ matrix
        transformed_matrices.append(transformed_matrix)
    
    return transformed_matrices

def validate_transformation_matrix(M):
    """
    Validate that a 4x4 matrix is a proper SE(3) transformation.
    """
    # Check bottom row
    expected_bottom = np.array([0, 0, 0, 1])
    if not np.allclose(M[3, :], expected_bottom, atol=1e-6):
        print(f"Warning: Bottom row is not [0 0 0 1]: {M[3, :]}")
    
    # Check rotation matrix orthogonality
    R = M[:3, :3]
    should_be_identity = R @ R.T
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        print(f"Warning: Rotation matrix is not orthogonal")
    
    # Check determinant
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        print(f"Warning: Rotation matrix determinant is {det}, should be 1.0")

def transform_original_extrinsic_to_inverse(M):
    """
    Compute the inverse of a 4x4 transformation matrix.
    For SE(3) matrices: T^(-1) = [R^T | -R^T * t; 0 0 0 1]
    """
    R = M[:3, :3]  # Extract rotation part
    t = M[:3, 3]   # Extract translation part
    
    # Compute inverse
    R_inv = R.T
    t_inv = -R_inv @ t
    
    # Build inverse matrix
    M_inv = np.eye(4)
    M_inv[:3, :3] = R_inv
    M_inv[:3, 3] = t_inv
    
    return M_inv

def parse_kitti_line_to_matrix(values):
    """
    Convert 16 values from KITTI format to 4x4 transformation matrix.
    """
    if len(values) != 16:
        raise ValueError(f"Expected 16 values, got {len(values)}")
    
    return np.array([
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5], values[6], values[7]],
        [values[8], values[9], values[10], values[11]],
        [values[12], values[13], values[14], values[15]]
    ])

def kitti_to_tum(input_file, output_file):
    """
    Convert KITTI trajectory file to TUM format with first frame as origin.
    """
    # First pass: read all matrices
    matrices = []
    with open(input_file, 'r') as f_in:
        for line_num, line in enumerate(f_in):
            values = list(map(float, line.strip().split()))
            
            if len(values) != 16:
                print(f"Warning: Line {line_num + 1} has {len(values)} values instead of 16")
                continue
            
            matrix = parse_kitti_line_to_matrix(values)
            matrices.append(matrix)
    
    if not matrices:
        raise ValueError("No valid matrices found in input file")
    
    # Get inverse of first matrix to set it as origin
    first_matrix_inv = transform_original_extrinsic_to_inverse(matrices[0])
    
    # Transform all matrices relative to first frame
    with open(output_file, 'w') as f_out:
        # Write TUM header
        f_out.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for timestamp, matrix in enumerate(matrices):
            # Apply transformation: T_new = T_first^(-1) * T_current
            transformed_matrix = first_matrix_inv @ matrix
            
            # Extract translation
            tx, ty, tz = transformed_matrix[0, 3], transformed_matrix[1, 3], transformed_matrix[2, 3]
            
            # Extract rotation matrix
            R = transformed_matrix[:3, :3]
            
            # Convert rotation matrix to quaternion
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
            
            # Write TUM format line
            f_out.write(f"{timestamp} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"Conversion complete! Output saved to {output_file}")
    print(f"First frame set as origin: [0.0 0.0 0.0 0.0 0.0 0.0 1.0]")
    print(f"This matches your results.txt format where first keyframe starts from origin.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python kitti_to_tum.py <input_file> <output_file>")
        print("Example: python kitti_to_tum.py traj.txt traj_tum.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        kitti_to_tum(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()