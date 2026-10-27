% Simulate a supervisory control system (Hybrid dynamics)

%Constants
Tf=10; %Final simulation time
h=1; %Sampling time for controller
U=2; %Control input absolute values
yo=[0.2, 1]; %I.C. (note second entry is initial input),

%Define smooth dynamics with events

%Define Event condition (as crossing through zero)
satEvent = @(t,y) (abs(y(1))-1)*mod(t,h); %Event of state crossing 1 or -1

E = odeEvent(EventFcn=satEvent, ...
             Direction="both", ...
             Response="callback", ...
             CallbackFcn=@satResponse);  %satResponse is the jump map

F=ode; %Define ODE object
F.ODEFcn=@myFlow; %Flow map
F.EventDefinition=E; %Events

F.InitialValue = yo; % we start in mode 1.

% Initial values on iteration
y=yo;
Tfi=0;
T=0;
To=0;
TotalSolution=[];
TotalTime=[];
EventTimes=[];
EventSolutions=[];

while  Tfi<Tf %Simulated time smaller than Tf

    F.InitialValue = y;
    S = solve(F,0,h-T); %Solve up to controller sampling time stop=true
    
    T=S.Time(end); %Time at which simulation stopped now, can be before h due to events
    y=S.Solution(:,end);
   
    %If stop=true
    TotalTime=[TotalTime To+S.Time];
    EventTimes=[EventTimes To+S.EventTime]; 
    To=To+T; %Total time simulated;

    TotalSolution=[TotalSolution S.Solution];
    EventSolutions=[EventSolutions S.EventSolution];
     
    %controller logic to be triggered when at multiples of h
    if mod(To,h)==0 %when stop=true this is needed to only execute periodically
       if y(1)>=1
            y(2)=-U;
        elseif y(1)<=-1
            y(2)=U;
        else
            y(2)=0;
        end
        Tfi=Tfi+h;
        T=0;
    end 
    
end
    
%Plotting
plot(TotalTime,TotalSolution);
hold on;
plot(EventTimes,EventSolutions,"ro"); %Plotting events as a red circle
hold off;

%Define jump map, in this case identity
function [stop,y] = satResponse(t,y)
    stop = true;
    y(1) = y(1);
    y(2) = y(2);
end