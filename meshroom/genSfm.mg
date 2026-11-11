{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "GenSfM": "1.0",
            "MeshFiltering": "3.0",
            "PrepareDenseScene": "3.1",
            "Publish": "1.3",
            "Texturing": "6.0"
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
            "inputs": {}
        },
        "GenSfM_1": {
            "nodeType": "GenSfM",
            "position": [
                220,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}"
            }
        },
        "PrepareDenseScene_1": {
            "nodeType": "PrepareDenseScene",
            "position": [
                460,
                -80
            ],
            "inputs": {
                "input": "{GenSfM_1.outputSfMData}"
            }
        },
        "MeshFiltering_1": {
            "nodeType": "MeshFiltering",
            "position": [
                460,
                80
            ],
            "inputs": {
                "inputMesh": "{GenSfM_1.outputMeshObj}",
                "keepLargestMeshOnly": false,
                "smoothingIterations": 1
            }
        },
        "Texturing_1": {
            "nodeType": "Texturing",
            "position": [
                700,
                0
            ],
            "inputs": {
                "input": "{GenSfM_1.outputSfMData}",
                "imagesFolder": "{PrepareDenseScene_1.output}",
                "inputMesh": "{MeshFiltering_1.outputMesh}",
                "textureSide": 4096,
                "downscale": 1,
                "colorMapping": {
                    "enable": true,
                    "colorMappingFileType": "png"
                },
                "fillHoles": true
            }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                940,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{Texturing_1.outputMesh}",
                    "{Texturing_1.outputMaterial}",
                    "{Texturing_1.outputTextures}",
                    "{GenSfM_1.outputMeshGlb}",
                    "{GenSfM_1.outputGaussianPly}",
                    "{GenSfM_1.outputSfMData}"
                ]
            }
        }
    }
}
