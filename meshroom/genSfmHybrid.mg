{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "DepthMap": "5.0",
            "DepthMapFilter": "4.0",
            "GenSfM": "1.0",
            "MeshFiltering": "3.0",
            "Meshing": "7.0",
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
            "inputs": {},
            "internalInputs": {
                "invalidation": "",
                "comment": "Input photos from random viewpoints.",
                "label": "CameraInit",
                "color": ""
            }
        },
        "GenSfM_1": {
            "nodeType": "GenSfM",
            "position": [
                240,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "generateSyntheticViews": 24
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "AI Core Reconstructor: Generates direct 3D mesh, sparse cameras, and dense orbital synthetic views.",
                "label": "GenSfM",
                "color": "#2b7bba"
            }
        },
        "PrepareDenseScene_Fast": {
            "nodeType": "PrepareDenseScene",
            "position": [
                520,
                -220
            ],
            "inputs": {
                "input": "{GenSfM_1.outputSfMData}"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Fast Branch: Undistorts original 2-4 input photos into EXR for direct multi-band texturing.",
                "label": "PrepareDenseScene_Fast",
                "color": "#2ca02c"
            }
        },
        "MeshFiltering_Fast": {
            "nodeType": "MeshFiltering",
            "position": [
                520,
                -80
            ],
            "inputs": {
                "inputMesh": "{GenSfM_1.outputMeshObj}",
                "keepLargestMeshOnly": false,
                "smoothingIterations": 1
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Fast Branch: Applies Laplacian surface smoothing (5 iterations) to the direct GenSfM AI 3D mesh.",
                "label": "MeshFiltering_Fast",
                "color": "#2ca02c"
            }
        },
        "Texturing_Fast": {
            "nodeType": "Texturing",
            "position": [
                800,
                -150
            ],
            "inputs": {
                "input": "{GenSfM_1.outputSfMData}",
                "imagesFolder": "{PrepareDenseScene_Fast.output}",
                "inputMesh": "{MeshFiltering_Fast.outputMesh}",
                "textureSide": 4096,
                "downscale": 1,
                "colorMapping": {
                    "enable": true,
                    "colorMappingFileType": "png"
                },
                "fillHoles": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Fast Direct AI Polish Branch - 4096px Multi-Band UV Texture.",
                "label": "Texturing_Fast",
                "color": "#2ca02c"
            }
        },
        "PrepareDenseScene_Dense": {
            "nodeType": "PrepareDenseScene",
            "position": [
                520,
                160
            ],
            "inputs": {
                "input": "{GenSfM_1.outputSfMDataDense}"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Dense Branch: Undistorts original views and synthetic 360-degree orbital views.",
                "label": "PrepareDenseScene_Dense",
                "color": "#ff7f0e"
            }
        },
        "DepthMap_Dense": {
            "nodeType": "DepthMap",
            "position": [
                760,
                160
            ],
            "inputs": {
                "input": "{PrepareDenseScene_Dense.input}",
                "imagesFolder": "{PrepareDenseScene_Dense.output}"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Dense Branch: CUDA PatchMatch Semi-Global Matching depth estimation across 24+ synthetic & real views.",
                "label": "DepthMap_Dense",
                "color": "#ff7f0e"
            }
        },
        "DepthMapFilter_Dense": {
            "nodeType": "DepthMapFilter",
            "position": [
                1000,
                160
            ],
            "inputs": {
                "input": "{DepthMap_Dense.input}",
                "depthMapsFolder": "{DepthMap_Dense.output}"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Dense Branch: Enforces multi-view depth consistency across neighboring orbital views.",
                "label": "DepthMapFilter_Dense",
                "color": "#ff7f0e"
            }
        },
        "Meshing_Dense": {
            "nodeType": "Meshing",
            "position": [
                1240,
                160
            ],
            "inputs": {
                "input": "{DepthMapFilter_Dense.input}",
                "depthMapsFolder": "{DepthMapFilter_Dense.output}"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Dense Branch: 3D Delaunay tetrahedralization and graph-cut surface extraction from filtered depth maps.",
                "label": "Meshing_Dense",
                "color": "#ff7f0e"
            }
        },
        "MeshFiltering_Dense": {
            "nodeType": "MeshFiltering",
            "position": [
                1480,
                160
            ],
            "inputs": {
                "inputMesh": "{Meshing_Dense.outputMesh}",
                "keepLargestMeshOnly": false,
                "smoothingIterations": 1
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Dense Branch: Cleans stray triangles and smooths dense MVS surface geometry.",
                "label": "MeshFiltering_Dense",
                "color": "#ff7f0e"
            }
        },
        "Texturing_Dense": {
            "nodeType": "Texturing",
            "position": [
                1720,
                160
            ],
            "inputs": {
                "input": "{Meshing_Dense.output}",
                "imagesFolder": "{DepthMap_Dense.imagesFolder}",
                "inputMesh": "{MeshFiltering_Dense.outputMesh}",
                "textureSide": 4096,
                "downscale": 1,
                "colorMapping": {
                    "enable": true,
                    "colorMappingFileType": "png"
                },
                "fillHoles": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Dense Sparse-to-Dense MVS Photogrammetry Branch - Multi-View UV Texture.",
                "label": "Texturing_Dense",
                "color": "#ff7f0e"
            }
        },
        "Publish_Comparison": {
            "nodeType": "Publish",
            "position": [
                2000,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{GenSfM_1.outputMeshObj}",
                    "{GenSfM_1.outputMeshMtl}",
                    "{GenSfM_1.outputMeshTexture}",
                    "{GenSfM_1.outputMeshGlb}",
                    "{GenSfM_1.outputGaussianPly}",
                    "{GenSfM_1.outputSfMData}",
                    "{Texturing_Fast.outputMesh}",
                    "{Texturing_Fast.outputMaterial}",
                    "{Texturing_Fast.outputTextures}",
                    "{Texturing_Dense.outputMesh}",
                    "{Texturing_Dense.outputMaterial}",
                    "{Texturing_Dense.outputTextures}",
                    "{GenSfM_1.outputSfMDataDense}"
                ]
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Comparison Publisher: Collects native textured mesh, GLB, and 3D Gaussian Splats alongside Texturing_Fast and Texturing_Dense.",
                "label": "Publish_Comparison",
                "color": "#9467bd"
            }
        }
    }
}
