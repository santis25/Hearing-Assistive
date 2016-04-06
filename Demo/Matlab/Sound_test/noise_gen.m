%adding noise to sound wave files

%{
for n = 1:1:100
    %load file
    [y,Fs] = audioread('School_Fire_Alarm.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('School_Fire_Vars/School_Wav_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end

%Other methods of adding noise
z = imnoise( y, 'salt & pepper', 0.05);
z = imnoise( y, 'poisson');
z = imnoise( y, 'gaussian', Mean, Var);
%}
for n = 1:1:100
    %load file
    [y,Fs] = audioread('Warning_Siren.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('Warning_Siren_Vars/Warning_Siren_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end

for n = 1:1:100
    %load file
    [y,Fs] = audioread('Smoke_Alarm.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('Smoke_Alarm_Vars/Smoke_Alarm_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end


for n = 1:1:100
    %load file
    [y,Fs] = audioread('Dog_Kennel.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('Dog_Kennel_Vars/Dog_Kennel_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end


for n = 1:1:100
    %load file
    [y,Fs] = audioread('Cow_Moos.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('Cow_Moos_Vars/Cow_Moos_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end


for n = 1:1:100
    %load file
    [y,Fs] = audioread('Cat_Meows.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('Cat_Meows_Vars/Cat_Meows_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end