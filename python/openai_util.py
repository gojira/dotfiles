import os
import openai
import json

"""
A config is a dictionary of environment variables that will be set before the API call.

Example config for Azure (not all are required for OpenAI):
{
    'OPENAI_API_KEY': 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'OPENAI_API_BASE': 'https://api.openai.com',
    'OPENAI_API_TYPE': 'azure',
    'OPENAI_API_VERSION': '2022-12-01',
    'OPENAI_ORGANIZATION': 'org-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
}

This works best if:
1. The environment variables are not already set
2. OR the environment variables only have the API key set.

The config will override any values currently set in the environment.  However, if the environment already has the variables set then the 
overrides will not work unless the environment variables are unset first.

This is because the openai library uses the environment variables to set the api_key, api_base, api_type, api_version, and organization and
the openai library doesn't have a way to unset these variables in place in a call.  For example, if organization is set in the environment
and openai library already has it initialized, passing e.g. `organization=None` or `organization=''` to the openai.Completion.create() call
does not work because the openai library will use the value that is already set in the environment.  The only way to unset the organization
is to unset the environment variable first. 

"""

def openai_params_from_config(params, config):
    """Initialize or update dictioary of OpenAI settings with values from config"""
    config_mapping = {
        'OPENAI_API_KEY': 'api_key',
        'OPENAI_API_BASE': 'api_base',
        'OPENAI_API_TYPE': 'api_type',
        'OPENAI_API_VERSION': 'api_version',
        'OPENAI_ORGANIZATION': 'organization',
    }
    if not params:
        params = { }
    if config:
        for config_key, param_key in config_mapping.items():
            if config_key in config:
                params[param_key] = config[config_key]
    return params


def get_completion_cli(prompt, config=None, **kwargs):
    """
    Get completion from OpenAI API

    """
    kwargs = openai_params_from_config(kwargs, config)
    if 'debug' in kwargs:
        del kwargs['debug']
        print(kwargs)

    response = openai.Completion.create(
        prompt=prompt,
        **kwargs
    )
    completion = response['choices'][0]['text']
    return completion


def get_embedding_cli(input, config=None, **kwargs):
    """
    Get embedding from OpenAI API

    """
    kwargs = openai_params_from_config(kwargs, config)
    if 'debug' in kwargs:
        del kwargs['debug']
        print(kwargs)

    response = openai.Embedding.create(
        input=input,
        **kwargs
    )
    embedding = response['data'][0]['embedding']
    return embedding

