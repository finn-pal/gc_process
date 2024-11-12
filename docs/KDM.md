# Katana Data Mover
## From my computer to Katana Scratch
```
me@localhost:~$ rsync -avh /path/to/my-directory z1234567@kdm.restech.unsw.edu.au:/srv/scratch/z1234567
```
## From Katana Scratch to my computer
```
me@localhost:~$ rsync -avh z1234567@kdm.restech.unsw.edu.au:/srv/scratch/my-remote-results /home/me
```

Dont forget about progress bar (i think -p, but check this)

/srv/scratch/astro/z5114326/gc_process

z5114326@kdm.restech.unsw.edu.au:/srv/scratch/astro/z5114326/gc_process/

/srv/scratch/astro/z5114326/gc_process/data

/Users/z5114326/Documents/GitHub/gc_process_katana/data/external