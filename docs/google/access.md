# Access and authentication

Access to Google Hardware is currently restricted to those in an approved
group.  In order to do this, you will need to apply for access, typically
in partnership with a Google sponsor.

Access and authentication for Google Hardware is done through Google Cloud
Platform.  In order to use Google Hardware through the Quantum Computing
Service, you will need a Google account and a Google Cloud Project.
There are two main categories of these accounts:

*   Personal accounts, such as a gmail.com account and a personally created
project.
*   Google enterprise (or Google Workspace) account.  These accounts are typically
centrally administered through an IT professional at your organization.
You may need to work with them in order to create projects, enable the API,
and do other steps in this guide, as those functions may be currently
restricted by your organization to prevent inadvertent costs or cloud-related
activity.

##  Create a project

You will need to use a Google Cloud Project to access Google Hardware though
the Quantum Computing Service.   If you have an existing project used for
Google Compute Engine, Cloud Storage, or other functionality, you can re-use
that project.  Alternatively, you can create a new project.

[Learn more about creating a project](https://cloud.google.com/docs/overview/)

Each project will have a project id that will be used in your code and
throughout these guides.

You do not need billing information to use the service at this time.

## API Access

You will need to configure your project to be able to access the API.

*   Log in and agree to Terms of Service
*   Follow this link to
[enable the Quantum Engine API](https://console.cloud.google.com/apis/library/quantum.googleapis.com?returnUrl=quantum)
in your Google Cloud Platform project.

If you are not able to enable the Quantum Engine API, then contact your Google
sponsor so that your project can be put on the approved list.

## User access

Each user (i.e. email address) will need to be associated with the project
and have appropriate IAM permissions on the project in order to access
the API through this project.

Each user will also have to be added to an approved list in order to access
the Quantum Computing Service.  Please submit any new users to your Google
sponsor so that they can be added.

## Next Steps

At this point, you should now have access to the Quantum Computing Service.
You can try out our [Getting Started Guide](../tutorials/google/start.ipynb).

You can also learn more about how to use the [Engine class](engine.md) to
access Google hardware or about [Google devices](devices.md) in the
documentation section for Google.
