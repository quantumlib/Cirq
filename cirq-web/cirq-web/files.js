/*
  Automatically retrieves files from the
  src folder so we're not continually updating
  the entry points.

  --Not finished
*/

const path = require('path');
const {readdir} = require('fs').promises;

const getSrcPaths = async function* (dir) {
  const readFlags = {withFileTypes: true};
  const firstLevel = await readdir(dir, readFlags);

  for (const file of firstLevel) {
    const res = path.resolve(dir, file.name);
    if (file.isDirectory()) {
      // yield* stops where you're at,
      // continues at designated generator obj
      yield* getSrcPaths(res);
    } else {
      yield res;
    }
  }
};

const generateEntryPoints = async function () {
  const res = {};
  for await (const file of getSrcPaths('./src')) {
    const fileName = file.split('/').pop();
    const key = fileName.slice(0, -3); // Remove .ts extension
    res[key] = `${fileName}`;
  }
  return res;
};

const resolvePromise = function () {
  generateEntryPoints().then(value => {
    return value;
  });
};

module.exports = {
  entry: resolvePromise(),
};

console.log(module.exports);
