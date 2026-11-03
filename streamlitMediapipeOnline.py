from src.AppRunner import runApp

appSources = [
    "User Image",
    "Local Image",
    "Online Image",
    "Video Sources",
]

runApp(
    appSources=appSources,
    idx_current_page=0,
    idx_current_module=0,
    idx_data_source=2,
    reset_on_reload=False,
    show_gc=True,
)
