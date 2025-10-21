from typing import Any, Callable


class NotifyHandler:
    events: dict = {}

    def handleEvent(self, caller: Any, eventName: str):
        fullName = type(caller).__name__ + '-' + eventName
        if fullName in self.events:
            self.events[fullName](caller)


# Create a global singleton when this module is imported.
if not 'notifyHandler' in globals():
    notifyHandler = NotifyHandler()
