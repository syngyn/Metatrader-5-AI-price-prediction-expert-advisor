; GGTHPredictorInstaller_SIMPLE.iss
; Simplified installer - guaranteed to compile
; Author: Jason Rusk

#define MyAppName "GGTH Predictor"
#define MyAppVersion "1.2"
#define MyAppPublisher "Jason Rusk"

[Setup]
AppId={{5C3A4F0B-2F0C-4E1A-8FA1-1F6D9A1A1234}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={userdocs}\GGTH Predictor
DefaultGroupName=GGTH Predictor
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=GGTHPredictorSetup_Simple
Compression=lzma
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install {#MyAppName} on your computer.%n%nIMPORTANT: Python 3.9-3.11 must be installed first.

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"

[Files]
Source: "ggthpredictor.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "config_manager.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "ggth_gui.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_ggth_gui.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_wizard.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "GGTH_User_Guide.docx"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "PredictionFinal.mq5"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "PredictionFinal.ex5"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Dirs]
Name: "{app}\models"

[Icons]
Name: "{group}\GGTH Predictor"; Filename: "{app}\run_ggth_gui.bat"; WorkingDir: "{app}"
Name: "{autodesktop}\GGTH Predictor"; Filename: "{app}\run_ggth_gui.bat"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\Configure MT5 Path"; Filename: "{app}\setup_wizard.bat"; WorkingDir: "{app}"

[Run]
Filename: "{app}\setup_wizard.bat"; Description: "Configure MT5 path now"; Flags: postinstall nowait skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  if not Exec('cmd.exe', '/c python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
  begin
    if MsgBox('Python not detected. Install Python 3.9-3.11 and add to PATH. Continue anyway?', mbConfirmation, MB_YESNO) = IDYES then
      Result := True
    else
      Result := False;
  end;
end;
