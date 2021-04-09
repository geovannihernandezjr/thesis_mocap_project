from dataclasses import dataclass
from dataclasses import field

@dataclass
class MoCapData:  # Struct like for storing summary of file content
    numOfFrames: int = 0
    numOfCameras: int = 0
    numOfMarkers: int = 0
    frequency: str = ""
    numOfAnalog: str = ""
    analogFrequency: str = ""
    description: str = ""
    timeStamp: list = field(default_factory=list)
    dataType: str = ""
    markerNames: list = field(default_factory=list)
    markerNamesXYZ: list = field(default_factory=list)
    totalNumOfMarkers: int = 0
    frameNumberList: list = field(default_factory=list)
    timeList: list = field(default_factory=list)
    # markerNamesXYZVel: list = field(default_factory=list)
    # markerNamesXYZAcc: list = field(default_factory=list)

