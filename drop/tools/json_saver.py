import sys
import logging
from typing import List, Optional, Tuple, Dict, TypeVar, Any, Union
import json
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path


def convert_DictConfig_to_dict_without_instantiation_args(dict_obj):
    """
       Converts a nested DictConfig or dictionary-like object into a standard Python dictionary,
       removing unnecessary attributes such as '_target_',  handling instantiated objects.

       Functionality:
       - If `dict_obj` is an instantiated object, it retrieves its attribute dictionary.
       - If `dict_obj` is a DictConfig, it is converted into a dictionary.
       - If a value within the dictionary is itself a DictConfig or ListConfig, it is recursively converted.
       - The key "_target_" is removed from the output.

       Args:
           dict_obj (Union[DictConfig, dict, object]): A nested dictionary-like object,
           possibly containing instantiated objects.

       Returns:
           Iterator[Tuple[str, Any]]: A generator yielding key-value pairs with transformed values.
    """

    if isinstance(dict_obj, DictConfig):
        dict_obj = OmegaConf.to_container(dict_obj)
    try:
        dict_obj = dict_obj.__dict__
    except:
        pass
    for key, value in dict_obj.items():
        try:
            if (
                str(value)[0] == "<"
            ):
                try:
                    value = value.__dict__  # if the object is instantiated, then get the attribute dict
                finally:
                    yield (key, dict(convert_DictConfig_to_dict_without_instantiation_args(value)))
        except:
            print(value)
        # Check if value is dict
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            yield (key, dict(convert_DictConfig_to_dict_without_instantiation_args(value)))

        else:
            if key not in ("_target_", "_partial_"):
                if isinstance(value, DictConfig) or isinstance(value, ListConfig):
                    value = OmegaConf.to_container(value)

                if type(value) == str:
                    if not value.startswith("functools.partial"):
                        yield (key, value)
                else:
                    yield (key, value)


class JsonSaver:
    def __init__(
        self,
        identifier: str,
        json_path: str,
    ) -> None:
        """
        A utility class for storing, updating, and retrieving structured JSON data,
        designed to facilitate searchability based on a specified identifier.

        JSON Structure:
            The JSON file consists of a list of dictionaries, where each dictionary entry
            contains an identifier and associated data.

        Args:
            identifier (str): The key used to search for entries in the JSON file (e.g., "epoch" or "criteria").
            json_path (str): The file path where the JSON data is stored.

        Methods:
            - set_out_path(out_path): Validates and sets the JSON storage path.
            - save_selected_data(identification_dict, data_name, data_dict): Adds or updates an entry in the JSON.
            - save_json_obj(json_object): Serializes and writes the JSON object to a file.
            - read_json(): Reads and returns the JSON file contents.
            - read_selected_data(params, data=None, params_name=None): Searches for a specific entry based on parameters.
            - remove_selected_data(identification_dict, data=None, identifier=None): Removes a specific entry from the JSON.

        Usage:
            >>> js = JsonSaver(identifier="epoch", json_path="results.json")
            >>> js.save_selected_data({"epoch": 10, "exclude_radiotherapy": True}, "accuracy", {"value": 0.85})
        """

        self.identifier = identifier
        self.json_path = None
        self.set_out_path(json_path)

    def set_out_path(self, out_path: str) -> None:
        if Path(out_path).suffix == ".json":
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(out_path).mkdir(parents=True, exist_ok=True)
        self.json_path = out_path

    def save_selected_data(self, identification_dict: Union[Dict, DictConfig], data_name: str, data_dict: Any) -> None:
        """
        Saves a data entry to the JSON file, ensuring proper structuring.

        Functionality:
          - If the JSON file does not exist, it is created with the provided data.
          - If an entry matching `identification_dict` is not found, a new entry is added.
          - If `identification_dict` is found, the new data is merged into the existing entry/ overwritten.

        Args:
          identification_dict (Union[Dict, DictConfig]): The identifier for the entry (e.g., model parameters).
          data_name (str): The key under which the provided data will be stored.
          data_dict (Any): The actual data to be saved.

        Side Effects:
          - Modifies the existing JSON file, adding new entries or updating existing ones.

        Example:
          >>> js.save_selected_data({"epoch": 5}, "loss", {"value": 0.2})
        """

        identification_dict = dict(convert_DictConfig_to_dict_without_instantiation_args(identification_dict))
        if self.read_json() == False:
            json_object = [{self.identifier: identification_dict, data_name: data_dict}]
            print("creating json")
        elif self.read_selected_data(identification_dict) == False:
            json_object = self.read_json()
            add_entry = {self.identifier: identification_dict, data_name: data_dict}
            json_object.append(add_entry)
            print("adding new criteria and first entry")
        else:
            json_object = self.read_json()
            sel_entry, index = self.read_selected_data(identification_dict)
            sel_entry[data_name] = data_dict
            json_object[index] = sel_entry
            print("adding new entry " + data_name)
        self.save_json_obj(json_object)

    def save_json_obj(self, json_object):
        # Serializing json
        json_object = json.dumps(json_object, indent=4)
        # Writing to sample.json
        with open(self.json_path, "w") as outfile:
            outfile.write(json_object)
        print("Saved to json")

    def read_json(self):
        """Reads json file and loads it."""
        try:
            with open(self.json_path) as fp:
                data = fp.read()
                data = json.loads(data)
            return data
        except:
            return False

    def read_selected_data(
        self, params: Union[Dict, DictConfig], data: Optional = None, params_name: Optional[str] = None
    ) -> Tuple[Dict[str, str], int]:

        """
        Retrieves a specific entry from the JSON file based on the given parameters.

        Functionality:
           - Iterates through the stored entries to find a match with `params`.
           - Returns the corresponding entry and its index in the JSON file.
           - If no matching entry is found, a warning is logged.

        Args:
           params (Union[Dict, DictConfig]): The search criteria.
           data (Optional[List[Dict]]): Preloaded JSON data, if available (default: None).
           params_name (Optional[str]): The key under which `params` should be searched (default: self.identifier).

        Returns:
           Tuple[Dict[str, str], int]: The found entry and its index in the JSON list.

        Raises:
           KeyError: If no matching entry is found.

        Example:
           >>> entry, index = js.read_selected_data({"epoch": 10})
        """

        params = dict(convert_DictConfig_to_dict_without_instantiation_args(params))
        if data is None:
            data = self.read_json()
        if params_name is None:
            params_name = self.identifier
        for i, entry in enumerate(data):
            if entry[params_name] == params:
                data_dict, index = entry, i
                break
        try:
            return data_dict, index
        except UnboundLocalError:
            logging.warning(
                f"The specified parameter dictionary does not exist in the data file.\n"
                f"Parameter dict: {params}\n"
                f"Data file: {self.json_path}\n"
                f"If this is unexpected, try running with `construct_dataset=True`."
            )
            return False

    def remove_selected_data(
        self, identification_dict: Union[Dict, DictConfig], data: Optional = None, identifier: Optional[str] = None
    ) -> None:

        """
        Removes a specific entry from the JSON file based on the provided identifier.

        Functionality:
            - Searches for an entry matching `identification_dict`.
            - If found, removes the entry and updates the JSON file.
            - Logs the removal operation.

        Args:
            identification_dict (Union[Dict, DictConfig]): The identifier for the entry to be removed.
            data (Optional[List[Dict]]): Preloaded JSON data, if available (default: None).
            identifier (Optional[str]): The key used for matching (default: self.identifier).

        Side Effects:
            - Modifies the existing JSON file by removing the specified entry.

        Example:
            >>> js.remove_selected_data({"epoch": 5})
        """

        identification_dict = dict(convert_DictConfig_to_dict_without_instantiation_args(identification_dict))
        if data is None:
            data = self.read_json()
        if identifier is None:
            identifier = self.identifier

        remove_index = None
        for i, x in enumerate(data):
            if x[identifier] == identification_dict:
                remove_index = i
                break
        if remove_index is not None:
            new_json = data.copy()
            del new_json[remove_index]
            logging.info(
                f"Entry {identifier} found at index {remove_index} in data file {self.json_path}. "
                f"Created a new json file without the specified entry (i.e. removed entry)."
            )
            self.save_json_obj(new_json)
        else:
            logging.info("No entry found/removed")
