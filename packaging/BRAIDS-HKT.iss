#ifndef Payload
  #define Payload "payload"
#endif
#ifndef Output
  #define Output "output"
#endif
[Setup]
AppId={{62B9501B-2D42-437B-A74F-665A9E382304}
AppName=BRAIDS-HKT
AppVersion=0.1.0
AppPublisher=BRAIDS-HKT
AppPublisherURL=https://github.com/cgotelli/BRAIDS-HKT
AppSupportURL=https://github.com/cgotelli/BRAIDS-HKT/issues
AppUpdatesURL=https://github.com/cgotelli/BRAIDS-HKT/releases
DefaultDirName={localappdata}\Programs\BRAIDS-HKT
DefaultGroupName=BRAIDS-HKT
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#Output}
OutputBaseFilename=BRAIDS-HKT-0.1.0-Windows-x64-Setup
Compression=lzma2/normal
LZMADictionarySize=65536
LZMANumBlockThreads=4
SolidCompression=yes
WizardStyle=modern
LicenseFile={#Payload}\LICENSE
UninstallDisplayIcon={app}\BRAIDS-HKT.ico
SetupIconFile={#Payload}\BRAIDS-HKT.ico
DisableProgramGroupPage=yes
CloseApplications=yes
RestartApplications=no
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: checkedonce
[Files]
Source: "{#Payload}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\BRAIDS-HKT"; Filename: "{app}\python\pythonw.exe"; Parameters: "-I -B ""{app}\launch.py"""; WorkingDir: "{app}"; IconFilename: "{app}\BRAIDS-HKT.ico"
Name: "{autodesktop}\BRAIDS-HKT"; Filename: "{app}\python\pythonw.exe"; Parameters: "-I -B ""{app}\launch.py"""; WorkingDir: "{app}"; IconFilename: "{app}\BRAIDS-HKT.ico"; Tasks: desktopicon
[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-I -B ""{app}\launch.py"""; WorkingDir: "{app}"; Description: "Launch BRAIDS-HKT"; Flags: nowait postinstall skipifsilent
