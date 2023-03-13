"""
* APIEndpoint class is the base abstract class that takes a config, an optional deployment, and optional logger
* OpenAIAPIEndpoint class is an APIEndpoint that uses OpenAI.  The primary difference is in the URL structure.
* AzureOpenAIAPIEndpoint class is an APIEndpoint that uses AzureOpenAI.  The URL structure is different from OpenAI, using the deployment name
* CLIAPI class uses openai-python to make the calls.  It passes api_key, api_type, api_base, and api_version from the config in the Completion and Embeddings calls.  There is no difference in the CLIAPI class between OpenAI and AzureOpenAI currently - as long as you call engine=deploymentname or engine=modelname interchangeably
* RESTAPI class uses REST to make the calls.  It uses the URL constructed in the OpenAIAPI or AzureOpenAIAPIclass 
"""

from abc import ABC, abstractmethod
import json
import requests
import openai
import os

def openai_params_from_config(params, config):
    """Initialize or update dictioary of OpenAI settings with values from config"""
    if not params:
        params = {
            'api_key': None,
            'api_base': None,
            'api_type': None,
            'api_version': None,
            'organization': None,
        }
    if config:
        if 'OPENAI_API_KEY' in config:
            params['api_key'] = config['OPENAI_API_KEY']
        if 'OPENAI_API_BASE' in config:
            params['api_base'] = config['OPENAI_API_BASE']
        if 'OPENAI_API_TYPE' in config:
            params['api_type'] = config['OPENAI_API_TYPE']
        if 'OPENAI_API_VERSION' in config:
            params['api_version'] = config['OPENAI_API_VERSION']
        if 'OPENAI_ORGANIZATION' in config:
            params['organization'] = config['OPENAI_ORGANIZATION']
    return params


class APIEndpoint(ABC):
    """
    Abstract base class for OpenAI API endpoints.

    Abstracts over the differences between the OpenAI and Azure APIs.
    """

    @abstractmethod
    def get_config(self, text: str) -> dict[str, str]:
        """Returns the configuration for the API."""
        raise NotImplementedError

    @abstractmethod
    def get_response(self, response_json: dict[str, str]) -> dict[str, str]:
        """Returns the response from the API."""
        raise NotImplementedError


class RESTAPIEndpoint(APIEndpoint):
    """
    Base class for OpenAI API endpoints that use the REST API.
    """
    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Returns the headers for the API."""
        raise NotImplementedError

    @abstractmethod
    def get_completion_url(self) -> str:
        """Returns the completion URL for the API."""
        raise NotImplementedError

    @abstractmethod
    def get_chat_completion_url(self) -> str:
        """Returns the chat completion URL for the API."""
        raise NotImplementedError

    @abstractmethod
    def get_embedding_url(self) -> str:
        """Returns the embedding URL for the API."""
        raise NotImplementedError

    @abstractmethod
    def get_completion_request(self, text: str) -> dict[str, str]:
        """Returns the completion request body for the API."""
        raise NotImplementedError

    @abstractmethod
    def get_embedding_request(self, text: str) -> dict[str, str]:
        """Returns the embedding request body for the API."""
        raise NotImplementedError


    def get_embedding(self, text: str) -> list[float]:
        request = self.get_request(text)
        headers = self.get_headers()
        url = self.get_url()

        with requests.Session() as session:
            try:
                response = session.post(url, json=request, headers=headers, verify=False)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"Request failed: {e}")
                print(f"Request: {request}")
                print(f"Headers: {headers}")
                print(f"URL: {url}")
                print(f"Response: {response.text}")
                raise e
            response_json = response.json()
            return response_json["data"][0]["embedding"]


    def get_completions(self, text: str) -> list[str]:
        request = self.get_request(text)
        headers = self.get_headers()
        url = self.get_url()

        with requests.Session() as session:
            try:
                response = session.post(url, json=request, headers=headers, verify=False)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"Request failed: {e}")
                print(f"Request: {request}")
                print(f"Headers: {headers}")
                print(f"URL: {url}")
                print(f"Response: {response.text}")
                raise e
            response_json = response.json()
            return response_json["data"][0]["completions"]


class OpenAIRESTEndpoint(RESTAPIEndpoint):
    def __init__(
        self,
        model: str,
        config: dict[str, str],
    ):
        assert config is not None, "config must be provided"
        assert config['api_key'] is not None, "api_key must be provided"
        assert config['api_type'] is not None, "api_type must be provided"
        assert model is not None, "model must be provided"

        super().__init__()

        self._config = config
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if config['organization'] is not None:
            self._headers["OpenAI-Organization"] = config['organization']

        self._model = model
        self._completion_url = openai.api_base + openai.api_version + "/completions"
        self._chat_completion_url = openai.api_base + openai.api_version + "/chat/completions"
        self._embedding_url = openai.api_base + openai.api_version + "/embeddings"

    def get_config(self) -> dict[str, str]:
        return self._config

    def get_headers(self) -> dict[str, str]:
        return self._headers

    def get_completion_url(self) -> str:
        return self._completion_url

    def get_chat_completion_url(self) -> str:
        return self._chat_completion_url

    def get_embedding_url(self) -> str:
        return self._embedding_url

    def get_completion_request(self, text: str, **kwargs) -> dict[str, str]:
        return {
            "prompt": text,
            "model": self._model,
            **kwargs,
        }

    def get_embedding_request(self, text: str, **kwargs) -> dict[str, str]:
        return {
            "input": text,
            "model": self._model,
            **kwargs,
        }


class AzureOpenAIEndpoint(APIEndpoint):
    def __init__(
        self,
        deployment_name: str,
        config: dict[str, str],
    ):
        assert config is not None, "config must be provided"
        assert config['api_key'] is not None, "api_key must be provided"
        assert config['api_type'] is not None, "api_type must be provided"
        assert config['api_version'] is not None, "api_version must be provided"
        assert config['api_base'] is not None, "api_base must be provided"
        assert deployment_name is not None, "deployment name must be provided"

        super().__init__()

        self._config = config
        self._headers = {
            "Content-Type": "application/json",
            "api-key": self._config['api_key'],
        }

        self._deployment_name = deployment_name
        self._completion_url = (
            f"{self._config['api_base']}/openai/deployments/{self._deployment_name}/"
            f"completions?api-version={self._config['api_version']}"
        )
        self._embedding_url = (
            f"{self._config['api_base']}/openai/deployments/{self._deployment_name}/"
            f"embeddings?api-version={self._config['api_version']}"
        )
        self.get_chat_completion_url = (
            f"{self._config['api_base']}/openai/deployments/{self._deployment_name}/"
            f"chat/completions?api-version={self._config['api_version']}"
        )

    def get_config(self) -> dict[str, str]:
        return self._config

    def get_headers(self) -> dict[str, str]:
        return self._headers

    def get_completion_url(self) -> str:
        return self._completion_url

    def get_chat_completion_url(self) -> str:
        return self._chat_completion_url

    def get_embedding_url(self) -> str:
        return self._embedding_url

    def get_completion_request(self, text: str, **kwargs) -> dict[str, str]:
        return {
            "prompt": text,
            #"model": self._model,
            **kwargs,
        }

    def get_embedding_request(self, text: str, **kwargs) -> dict[str, str]:
        return {
            "input": text,
            #"model": self._model,
            **kwargs,
        }


