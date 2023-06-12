from typing import Any, Callable


class NotifyHandler:
    events: dict = {}

    def addEvent(self, name: str, reaction: Callable[[Any], None]):
        self.events[name] = reaction

    def handleEvent(self, caller: Any, eventName: str):
        fullName = type(caller).__name__ + '-' + eventName
        if fullName in self.events:
            self.events[fullName](caller)


# Create a global singleton.
if not 'notifyHandler' in globals():
    notifyHandler = NotifyHandler()
