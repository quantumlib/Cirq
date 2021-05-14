## Cirq viz

### Scratch pad/things to do

- directory structure refactoring?
- **Testing, which framework?-- important**
    - [Mocha](https://mochajs.org/)
    - [Jasmine](https://jasmine.github.io/)
- add a server for live reloading/development purposes
    - maybe would be good to have a separate development directory so that ppl who want to develop can just go in there, spin up live-server, and do so.
    - [live-server](https://www.npmjs.com/package/live-server) via npm -- supports live reload
        - If we do go this route, might be better to start it from node rather than using the cmd line. We can add this to a starter script if we choose to make one.
    - Server should be run using nodejs for consistency, not python
- do we want some examples?
- do we want scripts to make it easier to run stuff? (ie. start-server.sh)
- **Compiling ts to js?** [This article talks about it here, specifically in vscode](https://code.visualstudio.com/docs/typescript/typescript-compiling). How will this affect live reloading?
- Formatting: [Prettier](prettier.io)

### Currently installed packages (see package-lock.json)
- [typescript](https://www.npmjs.com/package/typescript)
- [three.js](https://threejs.org) -- small note: minimized version for production, full for development

### Directions
Install packages with `npm install`.


