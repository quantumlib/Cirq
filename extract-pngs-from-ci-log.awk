1 { sub(/^[^ ]+ /, ""); }

/^%BEGIN / {
    filename = $2;
    pipe = "base64 --decode >" filename;
    print "writing", filename;
    next;
}

/^%END / {
    if (pipe != "")  close(pipe);
    pipe = "";
    print "done";
    next;
}

pipe != "" { print | pipe; }
