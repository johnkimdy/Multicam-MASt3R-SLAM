from torch.utils.data import Dataset
import bisect

class MultiDataset(Dataset):
    """
    Aggregates multiple MonocularDataset instances and returns synchronized frames based on 
    a reference camera. It is assumed that each MonocularDataset implements __getitem__(idx) 
    to return a tuple (timestamp, image).

    The reference camera is specified by reference_camera_id, and frames from the other datasets 
    are synchronized to the frame from the reference camera. If a dataset implements 
    get_item_by_timestamp(timestamp), that method is used; otherwise, the same index is used.
    """
    
    def __init__(self, datasets, camera_ids=None, reference_camera_id=None):
        """
        Args:
            datasets (list): List of MonocularDataset instances.
            camera_ids (list, optional): Unique camera IDs for each dataset.
                                         If not provided, they are assigned sequentially starting at 0.
            reference_camera_id (int, optional): The ID of the reference camera.
                                                 If not provided, the camera_id of the first dataset is used.
        """
        self.datasets = datasets
        
        # Assign camera_id to each dataset.
        if camera_ids is not None:
            if len(camera_ids) != len(datasets):
                raise ValueError("The lengths of datasets and camera_ids must match.")
            for ds, cid in zip(self.datasets, camera_ids):
                ds.camera_id = cid
        else:
            for i, ds in enumerate(self.datasets):
                ds.camera_id = i
        
        # Set the reference camera: use the provided reference_camera_id or default to the first dataset's camera_id.
        if reference_camera_id is None:
            self.reference_camera_id = self.datasets[0].camera_id
        else:
            self.reference_camera_id = reference_camera_id
        
        # Find the dataset corresponding to the reference camera.
        self.reference_dataset = None
        for ds in self.datasets:
            if ds.camera_id == self.reference_camera_id:
                self.reference_dataset = ds
                break
        if self.reference_dataset is None:
            raise ValueError(f"Reference camera ID {self.reference_camera_id} does not match any dataset's camera_id.")

    def __len__(self):
        # Use the length of the reference dataset.
        return len(self.reference_dataset)

    def __getitem__(self, idx):
        # Retrieve the frame from the reference dataset.
        ref_timestamp, ref_image = self.reference_dataset[idx]
        outputs = [(self.reference_dataset.camera_id, ref_timestamp, ref_image)]
        
        # For each other dataset, retrieve the frame synchronized to the reference timestamp.
        for ds in self.datasets:
            if ds is self.reference_dataset:
                continue
            if hasattr(ds, "get_item_by_timestamp"):
                ds_timestamp, ds_image = ds.get_item_by_timestamp(ref_timestamp)
            else:
                ds_timestamp, ds_image = ds[idx]
            outputs.append((ds.camera_id, ds_timestamp, ds_image))
        
        return outputs
    @property
    def reference(self):
        """
        Returns the reference dataset (i.e., the one with camera_id equal to reference_id).
        """
        return self.reference_dataset
    @property
    def datasets_by_camera(self):
        """
        Returns a dictionary that maps each camera_id to its corresponding dataset.
        This allows you to access a dataset using its camera id, for example:
            dataset = multi_dataset.datasets_by_camera[1]
        """
        return {ds.camera_id: ds for ds in self.datasets}