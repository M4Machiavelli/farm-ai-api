// metro.config.js
const { getDefaultConfig } = require('@react-native/metro-config');

const config = getDefaultConfig(__dirname);

module.exports = config;
/**
 * Metro configuration for React Native
 * https://facebook.github.io/metro/docs/configuration
 *
 * @format
 */

module.exports = {
  transformer: {
    assetPlugins: ['react-native-svg-asset-plugin'],
  },
  resolver: {
    sourceExts: ['jsx', 'js', 'ts', 'tsx'], // Add other extensions if needed
  },
};

