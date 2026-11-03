from src.AppRunner import runApp

appSources = [
    "User Image",
    "Local Image",
    "Online Image",
    "User Video",
    "Local Video",
    "Online Video",
    "WebCam",
]

runApp(
    appSources=appSources,
    idx_current_page=1,
    idx_current_module=2,
    idx_data_source=1,
    reset_on_reload=True,
)
