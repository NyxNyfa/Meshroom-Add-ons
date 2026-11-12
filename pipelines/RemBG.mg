{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "RemBG": "1.0",
            "Publish": "1.3"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                0,
                0
            ],
            "inputs": {
                "allowSingleView": 1
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Input photos for automatic AI background removal.",
                "label": "CameraInit",
                "color": ""
            }
        },
        "RemBG_1": {
            "nodeType": "RemBG",
            "position": [
                300,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "model": "u2net",
                "executionBackend": "local_wsl_gpu",
                "alphaMatting": false,
                "padToSquare": false
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "AI-Powered Automatic Background Removal (Transparent RGBA & Grayscale Masks).",
                "label": "RemBG",
                "color": "#1f77b4"
            }
        },
        "Publish_Outputs": {
            "nodeType": "Publish",
            "position": [
                620,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{RemBG_1.outputFolder}",
                    "{RemBG_1.outputMasks}"
                ],
                "output": "{projectFolder}/Published_RemBG"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Exports segmented transparent RGBA photos and masks.",
                "label": "Publish_Outputs",
                "color": "#2ca02c"
            }
        }
    }
}

