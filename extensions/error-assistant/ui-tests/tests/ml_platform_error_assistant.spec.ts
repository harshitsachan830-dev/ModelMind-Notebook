import { expect, test } from '@jupyterlab/galata';

/**
 * Don't load JupyterLab webpage before running the tests.
 * This is required to ensure we capture all log messages.
 */
test.use({ autoGoto: false });

test('hides Help and adds an Error Assistant command to the View menu', async ({
  page
}) => {
  await page.goto();
  await expect(page.getByRole('menuitem', { name: 'Help' })).toHaveCount(0);
  await page.getByRole('menuitem', { name: 'View' }).click();
  await expect(
    page.getByRole('menuitem', { name: 'Open Error Assistant' })
  ).toBeVisible();
  await page.getByRole('menuitem', { name: 'Open Error Assistant' }).click();
  await expect(page.getByText('Platform service: Ready')).toBeVisible();
});
