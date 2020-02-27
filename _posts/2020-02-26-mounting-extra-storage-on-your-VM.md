# How to mount an additional storage disk on your azure VM

For this tutorial, I owe huge thanks to my fellow OpenAI Scholar, [Alethea](https://aletheap.github.io/), who taught me how to do this. This is a writeup of Alethea's in-person tutorial for me.

I will assume that you have an [azure virtual machine (VM)](https://azure.microsoft.com/en-us/services/virtual-machines/) already set up, and that you're finding that you don't have enough disk space. I also assume that you chose an Ubuntu (Linux) VM.
(If you're just getting started with virtual machines, I recommend a [preconfigured data science machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/))

1. In a web browser, open your azure portal and go to your VM.
    1. "Stop" your VM, and wait until the machine is stopped.
    1. Select "Disks" from the panel on the left.
        ![](/images/mounting/s1-select-disks.png "S1")
        1. Select "+ Add data disk".
        1. Under the "Name" dropdown menu, select "Create disk".
            ![](/images/mounting/s2-create-disk.png "S2")
            1. Give it a name.
            1. Change the size to fit your needs. I recommend 2048 GB = 2T.
            1. (You can leave the rest as the defaults.)
            1. Select "Create". (...and wait).
            1. Select "Save". (...and wait).
    1. Return to your VM (by selecting "Overview" in the left panel), and select "Start".  (It might help to refresh the page, so you can see once it loads). (...and wait).
        1. While you wait, you can observe your machine as it's starting up by selecting "Serial console" in the left-hand panel. (If your VM throws a tantrum in the future, the serial console can be your point of entry for logging in and rescuing your work.)
        ![](/images/mounting/s3-serial-console.png "S3")


2. Go to terminal on your local computer, and ssh into your VM.
    1. In your VM terminal window, type `lsblk`.
        ![](/images/mounting/s4-lsblk.png "S4")
        1. Looking at the output of `lsblk`, think of each row as a hard drive. Search for the one (under `NAME`) that has the large storage (under `SIZE`) that you selected. In my case, it's `sdc` because I can see that it has `SIZE`, `2T`.
        1. You can now set a variable to help you keep track of your disk name: `BIGDISK="sdc"`.
    1. Now pay attention, because if you get this wrong, you might destroy your VM (but no pressure ofc):
        - In your VM terminal, type: `sudo fdisk /dev/$BIGDISK`
        - In **my** case, this will be: `sudo fdisk /dev/sdc`
        ![](/images/mounting/s5-fdisk.png "S5")
    1. In the fdisk program (after `Command (m for help):`), type:
        1. `g` (This creates a [partition table](https://en.wikipedia.org/wiki/Partition_table) on your disk.)
        1. `n` (This creates a partition.)
            - You will be asked three questions: Hit `Enter` for each of them to select the default option.
            - It will confirm that it created a partition for you (a single partition for the entire hard drive).
        1. `w` (This applies all the commands above to the disk you created).
            - The fdisk program now finishes and returns you to your VM terminal.
            - You now have a partition table on the large hard drive that you created.   

    1. Back in your VM terminal, you now want to create a [file system](https://en.wikipedia.org/wiki/File_system) on your partition:
        - `sudo mkfs -t ext4 -j -L bigdisk /dev/${BIGDISK}1`
            - Don't forget the 1 at the end
        - In **my** case, this would be: `sudo mkfs -t ext4 -j -L bigdisk /dev/sdc1`
            - The `mkfs` command **m**a**k**es the **f**ile **s**ystem.
            - `ext4` is the default file system for Linux.
            - `-L` adds metadata to the file system: It gives a persistent handle for automatic mounting.
                - Our handle is `bigdisk`. We will refer back to it later.
    1. Now we will create a directory, where we will [mount](https://en.wikipedia.org/wiki/Mount_(computing)) the disk. In your VM terminal, type:
        - `sudo mkdir /largedisk`
    1. Next, we will edit the [`fstab`](https://help.ubuntu.com/community/Fstab) configuration file: This will make our disk automatically mount to our preferred location (`/largedisk`) in our file system whenever you boot your VM. (Trust me, you don't want to repeat this process every morning.) In your VM terminal, type:
        - `sudo nano /etc/fstab`
            - [`nano`](https://en.wikipedia.org/wiki/GNU_nano) is a text editor that allows you to look at the `fstab` file and add a line to it.
        ![](/images/mounting/s6-edit-fstab.png "S6")
        - In the `fstab` file, which you are accessing through the `nano` text editor, add the following words, separated by space(s), all on a single line:
            - `LABEL=bigdisk    /largedisk    ext4    defaults,rw    1    2`
                - `LABEL` means: Look for a file system that has this label. Recall that we set the label to `bigdisk` previously.
                - `/largedisk` is where we want to mount our partition.
                - `defaults,rw` means: Keep all the defaults, but make sure that we have permission to read and write to this directory.
                - `1   2` are flags that tell Ubuntu to automatically mount the partition upon booting.
        - Press `Ctrl`+`o` on your keyboard to [tell nano to save the file](https://wiki.gentoo.org/wiki/Nano/Basics_Guide#Saving_and_exiting).
            - It will ask you if the file name is OK: Press `Enter`.
        - Press `Ctrl`+`x` to exit nano. This will return you to your VM terminal.

1. Reboot the VM.
    - Exit your VM from your VM terminal (by typing `exit`). You are now in your local computer's terminal.
    - Go to your azure portal in a web browser, and navigate to your VM. Select "Restart".
        - You can again observe what your VM is up to in the "Serial Console".
    - ssh back in from your local machine's terminal.
    - (The reason why we did this is because we're going to change the owner of the big disk that you created. Restarting makes the disk actually mount to the correct directory. If the disk is not mounted, you would change the owner just of the directory on your smaller original drive, not of the big storage disk that you created.)

1. In your VM terminal, type:
    - `mount | grep largedisk`
        - You should now see an output like this:
        ![](/images/mounting/s7-mount-grep.png "S7")
    - `df -h | grep largedisk`
        - You should now see an output, which shows the (correct) amount of storage on your disk:
        ![](/images/mounting/s8-df-h-grep.png "S8")
    - `sudo chown $USER /largedisk`
        - This changes the owner of your big disk, so you can write to it without calling `sudo` all the time.
1. You are **DONE!** (If you like, you might do `touch /largedisk/success.txt` just for the satisfaction.) Happy saving.
