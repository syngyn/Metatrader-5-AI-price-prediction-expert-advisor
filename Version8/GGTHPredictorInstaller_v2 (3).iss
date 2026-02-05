; GGTHPredictorInstaller_v2.iss
; User-level installation with MT5 directory configuration
; Author: Jason Rusk
; Updated for GGTH Predictor v2.0 / unified_predictor_v8.1

#define MyAppName "GGTH Predictor"
#define MyAppVersion "2.0"
#define MyAppPublisher "Jason Rusk"
#define MyAppExeName "run_ggth_gui.bat"
#define PythonDownloadURL "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

[Setup]
AppId={{5C3A4F0B-2F0C-4E1A-8FA1-1F6D9A1A1234}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Install to user's Documents folder - no admin required
DefaultDirName={userdocs}\GGTH Predictor
DefaultGroupName=GGTH Predictor
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=GGTHPredictorSetup_v2
Compression=lzma
SolidCompression=yes
; No admin privileges required
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
; User-level uninstall info
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\run_ggth_gui.bat
; Version info
VersionInfoVersion=2.0.0.0
VersionInfoDescription=GGTH ML Forex Predictor
VersionInfoCopyright=Copyright (C) 2026 Jason Rusk

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install {#MyAppName} v{#MyAppVersion} on your computer.%n%nIncludes:%n• ML Prediction Engine (LSTM, GRU, Transformer, TCN, LightGBM)%n• Multi-timeframe predictions (1H, 4H, 1D)%n• Graphical User Interface%n• MetaTrader 5 Expert Advisor%n%nRequires Python 3.9-3.11 (will be checked).

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
Name: "installEA"; Description: "Install MetaTrader 5 Expert Advisor"; GroupDescription: "MT5 Integration:"; Flags: checkedonce

[Files]
; Core Python files
Source: "unified_predictor_v8.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "ggth_gui.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "config_manager.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Launcher scripts
Source: "run_ggth_gui.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_wizard.bat"; DestDir: "{app}"; Flags: ignoreversion

; Documentation
Source: "GGTH_Users_Guide.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "GGTH_Quick_Start.md"; DestDir: "{app}"; Flags: ignoreversion

; EA file: installed into user-specified MT5 directory (only if task selected)
Source: "PredictionFinal_v8_1.mq5"; DestDir: "{code:GetMT5ExpertsDir}"; Flags: ignoreversion; Tasks: installEA

[Dirs]
Name: "{code:GetMT5ExpertsDir}"; Tasks: installEA
Name: "{app}\models"
Name: "{app}\logs"

[Icons]
Name: "{group}\GGTH Predictor GUI"; Filename: "{app}\run_ggth_gui.bat"; WorkingDir: "{app}"; IconFilename: "{sys}\shell32.dll"; IconIndex: 165
Name: "{group}\Full User Guide"; Filename: "{app}\GGTH_Users_Guide.txt"
Name: "{group}\Setup Wizard"; Filename: "{app}\setup_wizard.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall GGTH Predictor"; Filename: "{uninstallexe}"
Name: "{autodesktop}\GGTH Predictor"; Filename: "{app}\run_ggth_gui.bat"; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{sys}\shell32.dll"; IconIndex: 165

[Run]
; Option to launch GUI after install
Filename: "{app}\run_ggth_gui.bat"; Description: "Launch GGTH Predictor GUI"; Flags: postinstall nowait skipifsilent unchecked; WorkingDir: "{app}"
; Option to view quick start guide
Filename: "{app}\GGTH_Users_Guide.txt"; Description: "View Quick Start Guide"; Flags: postinstall shellexec skipifsilent unchecked

[UninstallDelete]
; Clean up generated files on uninstall
Type: files; Name: "{app}\config.json"
Type: files; Name: "{app}\*.pyc"
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}\logs"
Type: dirifempty; Name: "{app}\__pycache__"
; Don't delete .venv - user may want to keep it

[Code]
var
  PythonDetected: Boolean;
  PythonVersion: String;
  PythonPath: String;
  MT5DirectoryPage: TInputDirWizardPage;
  MT5FilesPath: String;

{ Check if Python is installed and get version }
function IsPythonInstalled(): Boolean;
var
  ResultCode: Integer;
  TempFile, Output: String;
  OutputLines: TArrayOfString;
begin
  Result := False;
  PythonVersion := '';
  PythonPath := '';
  
  { Try to run python --version }
  if Exec('cmd.exe', '/c python --version 2>&1', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      Result := True;
      
      { Try to get the actual version string }
      TempFile := ExpandConstant('{tmp}\pyver.txt');
      if Exec('cmd.exe', '/c python --version > "' + TempFile + '" 2>&1', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      begin
        if LoadStringsFromFile(TempFile, OutputLines) then
        begin
          if GetArrayLength(OutputLines) > 0 then
            PythonVersion := OutputLines[0];
        end;
        DeleteFile(TempFile);
      end;
      
      if PythonVersion = '' then
        PythonVersion := 'Python (version unknown)';
    end;
  end;
end;

{ Attempt to auto-detect MT5 Files directory }
function AutoDetectMT5FilesPath(): String;
var
  Base, TerminalRoot, FilesPath: String;
  FindRec: TFindRec;
  FoundPaths: TStringList;
  i: Integer;
begin
  Result := '';
  FoundPaths := TStringList.Create;
  
  try
    { Try %APPDATA%\MetaQuotes\Terminal\<hash>\MQL5\Files }
    Base := ExpandConstant('{userappdata}') + '\MetaQuotes\Terminal';
    if DirExists(Base) then
    begin
      if FindFirst(Base + '\*', FindRec) then
      begin
        try
          repeat
            if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
            begin
              if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
              begin
                TerminalRoot := Base + '\' + FindRec.Name;
                FilesPath := TerminalRoot + '\MQL5\Files';
                if DirExists(FilesPath) then
                begin
                  FoundPaths.Add(FilesPath);
                end;
              end;
            end;
          until not FindNext(FindRec);
        finally
          FindClose(FindRec);
        end;
      end;
    end;
    
    { Return the most recently modified one (likely active installation) }
    if FoundPaths.Count > 0 then
    begin
      Result := FoundPaths[0];
      { If multiple found, we just use the first one - user can change it }
    end;
    
    { Fallback: default MetaTrader 5 installation in Program Files }
    if Result = '' then
    begin
      FilesPath := ExpandConstant('{autopf}') + '\MetaTrader 5\MQL5\Files';
      if DirExists(FilesPath) then
        Result := FilesPath;
    end;
    
  finally
    FoundPaths.Free;
  end;
end;

{ Initialize wizard pages }
procedure InitializeWizard();
var
  DefaultMT5Path: String;
begin
  { Create custom page for MT5 directory }
  MT5DirectoryPage := CreateInputDirPage(
    wpSelectDir,
    'Select MetaTrader 5 Files Directory',
    'Where is your MT5 MQL5\Files directory located?',
    'The GGTH Predictor saves prediction files to your MT5 Files directory ' +
    'so the Expert Advisor can read them.' + #13#10#13#10 +
    'Typical location:' + #13#10 +
    'C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\[HASH]\MQL5\Files' + #13#10#13#10 +
    'To find this manually:' + #13#10 +
    '1. Open MetaTrader 5' + #13#10 +
    '2. Click File → Open Data Folder' + #13#10 +
    '3. Navigate to MQL5\Files' + #13#10 +
    '4. Copy the path from the address bar' + #13#10#13#10 +
    'The installer has attempted to auto-detect this location below.',
    False,
    ''
  );

  { Try to auto-detect MT5 path }
  DefaultMT5Path := AutoDetectMT5FilesPath();
  
  if DefaultMT5Path <> '' then
    MT5DirectoryPage.Values[0] := DefaultMT5Path
  else
    MT5DirectoryPage.Values[0] := ExpandConstant('{userappdata}') + '\MetaQuotes\Terminal\';
end;

{ Initialize setup - check for Python }
function InitializeSetup(): Boolean;
var
  ErrorMsg: String;
  ResultCode: Integer;
begin
  Result := True;
  PythonDetected := IsPythonInstalled();
  
  if not PythonDetected then
  begin
    ErrorMsg := 'Python Not Detected!' + #13#10#13#10 +
                'GGTH Predictor requires Python 3.9, 3.10, or 3.11.' + #13#10#13#10 +
                'Please install Python first from:' + #13#10 +
                'https://www.python.org/downloads/' + #13#10#13#10 +
                '╔════════════════════════════════════════════════════╗' + #13#10 +
                '║  IMPORTANT: During Python installation, you MUST   ║' + #13#10 +
                '║  check the box: "Add Python to PATH"               ║' + #13#10 +
                '╚════════════════════════════════════════════════════╝' + #13#10#13#10 +
                'After installing Python, run this installer again.' + #13#10#13#10 +
                'Would you like to open the Python download page now?';
    
    if MsgBox(ErrorMsg, mbError, MB_YESNO) = IDYES then
    begin
      ShellExec('open', '{#PythonDownloadURL}', '', '', SW_SHOW, ewNoWait, ResultCode);
    end;
    
    Result := False;
  end
  else
  begin
    MsgBox('✓ Python Detected!' + #13#10#13#10 + 
           PythonVersion + #13#10#13#10 +
           'Installation will continue.' + #13#10#13#10 +
           'Note: First launch may take 5-10 minutes to download' + #13#10 +
           'and install required Python packages (TensorFlow, etc.).',
           mbInformation, MB_OK);
  end;
end;

{ Validate MT5 directory before continuing }
function NextButtonClick(CurPageID: Integer): Boolean;
var
  SelectedPath: String;
begin
  Result := True;
  
  { Validate MT5 directory page }
  if CurPageID = MT5DirectoryPage.ID then
  begin
    SelectedPath := MT5DirectoryPage.Values[0];
    
    { Check if directory exists }
    if not DirExists(SelectedPath) then
    begin
      MsgBox('The selected directory does not exist!' + #13#10#13#10 +
             'Please select a valid MetaTrader 5 Files directory.' + #13#10#13#10 +
             'How to find it:' + #13#10 +
             '1. Open MetaTrader 5' + #13#10 +
             '2. Click File → Open Data Folder' + #13#10 +
             '3. Navigate to MQL5\Files' + #13#10 +
             '4. Copy the path',
             mbError, MB_OK);
      Result := False;
      Exit;
    end;
    
    { Verify it looks like an MT5 Files directory }
    if (Pos('MQL5', SelectedPath) = 0) and (Pos('Files', SelectedPath) = 0) then
    begin
      if MsgBox('Warning: This directory does not appear to be an MQL5\Files folder.' + #13#10#13#10 +
                'Expected path format:' + #13#10 +
                '...\MetaQuotes\Terminal\[HASH]\MQL5\Files' + #13#10#13#10 +
                'If this is incorrect, the predictor will not be able to ' +
                'communicate with MetaTrader 5.' + #13#10#13#10 +
                'Continue anyway?',
                mbConfirmation, MB_YESNO) = IDNO then
      begin
        Result := False;
        Exit;
      end;
    end;
    
    { Store the path for later use }
    MT5FilesPath := SelectedPath;
  end;
end;

{ Get MT5 Experts directory for EA installation }
function GetMT5ExpertsDir(Param: String): String;
var
  ExpertsPath, MQL5Path: String;
begin
  { Get the parent directory of Files (should be MQL5), then navigate to Experts\GGTH }
  { RemoveBackslashUnlessRoot is the Inno Setup equivalent of ExcludeTrailingPathDelimiter }
  MQL5Path := ExtractFilePath(RemoveBackslashUnlessRoot(MT5FilesPath));
  ExpertsPath := MQL5Path + 'Experts\GGTH';
  Result := ExpertsPath;
end;

{ Escape backslashes for JSON }
function EscapePathForJSON(const Path: String): String;
var
  i: Integer;
begin
  Result := '';
  for i := 1 to Length(Path) do
  begin
    if Path[i] = '\' then
      Result := Result + '\\'
    else
      Result := Result + Path[i];
  end;
end;

{ After installation completes, create config file }
procedure CurStepChanged(CurStep: TSetupStep);
var
  ConfigFile: String;
  ConfigContent: TStringList;
  EscapedPath: String;
begin
  if CurStep = ssPostInstall then
  begin
    { Create config.json with MT5 path }
    ConfigFile := ExpandConstant('{app}') + '\config.json';
    EscapedPath := EscapePathForJSON(MT5FilesPath);
    
    ConfigContent := TStringList.Create;
    try
      ConfigContent.Add('{');
      ConfigContent.Add('  "mt5_files_path": "' + EscapedPath + '",');
      ConfigContent.Add('  "version": "2.0",');
      ConfigContent.Add('  "use_kalman": true,');
      ConfigContent.Add('  "default_symbol": "EURUSD",');
      ConfigContent.Add('  "prediction_interval_minutes": 60,');
      ConfigContent.Add('  "default_models": ["lstm", "transformer", "lgbm"],');
      ConfigContent.Add('  "available_models": ["lstm", "gru", "transformer", "tcn", "lgbm"],');
      ConfigContent.Add('  "installed": "' + GetDateTimeString('yyyy-mm-dd hh:nn:ss', #0, #0) + '"');
      ConfigContent.Add('}');
      ConfigContent.SaveToFile(ConfigFile);
    finally
      ConfigContent.Free;
    end;
  end;
end;

{ Show completion message }
procedure CurPageChanged(CurPageID: Integer);
var
  ExpertsDir: String;
begin
  if CurPageID = wpFinished then
  begin
    ExpertsDir := GetMT5ExpertsDir('');
    
    WizardForm.FinishedLabel.Caption := 
      'GGTH Predictor v2.0 has been installed successfully!' + #13#10#13#10 +
      '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' + #13#10 +
      'Installation Locations:' + #13#10 +
      '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' + #13#10 +
      'Predictor: ' + ExpandConstant('{app}') + #13#10 +
      'MT5 Files: ' + MT5FilesPath + #13#10 +
      'EA Location: ' + ExpertsDir + #13#10#13#10 +
      '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' + #13#10 +
      'Next Steps:' + #13#10 +
      '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' + #13#10 +
      '1. Launch GGTH Predictor from desktop/Start Menu' + #13#10 +
      '2. First launch takes 5-10 min (installs packages)' + #13#10 +
      '3. Train models for your currency pairs' + #13#10 +
      '4. In MT5: Compile and attach the EA to a chart' + #13#10#13#10 +
      'See the Quick Start Guide for detailed instructions.';
  end;
end;
